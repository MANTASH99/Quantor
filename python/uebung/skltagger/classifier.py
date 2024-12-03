"""
SKLTagger: Pseudo Markov classifier
"""
import numpy as np
import scipy as sp
import pandas as pd
import re

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_scalar, check_is_fitted, check_consistent_length
from sklearn.utils.multiclass import unique_labels

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

class PseudoMarkovClassifier(ClassifierMixin, BaseEstimator):
    """ Two-stage classifier that simulates an additional HMM layer with context tag probabilities.

    The first stage of the classifier is hard-coded to logistic regression trained by SGD. Since it is only
    used to predict tag probabilities for context tokens, it does not have to be tuned for optimal accuracy
    and hence need not be configurable by users.
    
    The second stage classifier is provided by the user and can be any multinomial classifier. A drawback of
    this approach is that meta-parameters have to be initialised in the classifier prototype object and cannot
    be fine-tuned with standard grid search. Alternatives would be to either hard-code the second-stage classifier
    algorithm and expose its key meta-parameters (a linear SVM is relatively slow but robustly achieves good 
    accuracy), to use SGD with the same meta-parameters for both stages, or to implement the first stage separately
    as a transformer that extends the feature matrix with context tag probabilities and can be combined with
    an arbitrary classifier in a pipeline.
    
    Parameters
    ----------
    clf : object
        Classifier object to be used as the second stage classifier, with all meta-parameters initialised.
        The object is taken as a prototype and cloned before use. Note that parallelisation of this classifier
        may need to be enabled explicitly even if the `n_jobs` argument is specified for the first stage.
    alpha : float, default=1e-6
        Regularisation strength of the first-stage classifier.
    max_iter : int, default=5000
        Maximal number of SGD epochs for the first-stage classifier.
    n_jobs : int, default=None
        Number of CPUs to use when training the first-stage classifier.
    left_context : int, default=2
        Number of tokens before the current token for which context tag probabilities are computed.
    right_context : int, default=2
        Number of tokens after the current token for which context tag probabilities are computed.

    Attributes
    ----------
    clf1_ : object
        Fitted first-stage classifier.
    clf2_ : object
        Fitted second-stage classifier.
    classes_ : array of shape (n_classes,)
        Categories (i.e. POS tags) assigned by the second-stage classifier.
    offsets_ : list of int
        Offsets for shifting the matrix of tag probabilities predicted by the first-stage classifier.

    **TODO** 
    """
    def __init__(self, clf, alpha=1e-6, max_iter=5000, n_jobs=None, 
                       left_context=2, right_context=2):
        self.clf = clf
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.left_context = left_context
        self.right_context = right_context

    ## public API
    def fit(self, X, y):
        """ Train both classifier stages on the provided data.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            Training feature matrix (without context tag probabilities).
        y : array of shape (n_samples,)
            Vector of target values (i.e. POS tags).

        Returns
        -------
        self : object
            Fitted classifier.
        """
        # check that input is well-formed
        X, y = check_X_y(X, y, accept_sparse='csr')
        
        # validate all parameter settings (must be done here, not in __init__)
        if not (isinstance(self.clf, BaseEstimator) and isinstance(self.clf, ClassifierMixin)):
            raise TypeError('second-stage classifier must be a Scikit-Learn classifier object')
        check_scalar(self.alpha, 'alpha', target_type=float, min_val=0.0)
        check_scalar(self.max_iter, 'max_iter', target_type=int, min_val=100)
        check_scalar(self.left_context, 'left_context', target_type=int, min_val=0, max_val=20)
        check_scalar(self.right_context, 'right_context', target_type=int, min_val=0, max_val=20)
        
        # fit first-stage classifier
        self.clf1_ = SGDClassifier(loss='log_loss', alpha=self.alpha, max_iter=self.max_iter, n_jobs=self.n_jobs)
        self.clf1_.fit(X, y)
        
        # compute tag probabilities
        # NB: It would be more sensible to do this via two-fold cross-validation, fitting the first-stage classifier
        # twice on two halves of the training data. Special precaution would have to be taken to ensure that exactly
        # the same tagset is used for each fold, e.g. by replacing very rare tags with OTHER (in the first stage only!). 
        # We take the lazy approach here and use the overfitted tag probabilities for training the second-stage classifier,
        # which doesn't seem to make much of a difference wrt. accuracy on the test set.
        X_prob = self.clf1_.predict_proba(X)
        
        # use circular shift to obtain context tag probabilities
        self.offsets_ = [ i for i in range(1, self.left_context + 1) ] + [ -i for i in range(1, self.right_context + 1) ]
        Xs_ctxt = [ np.roll(X_prob, offset, axis=0) for offset in self.offsets_ ]
        
        # combine with sparse feature matrix
        X_ext = sp.sparse.hstack([X] + Xs_ctxt, format='csr')
        
        # fit second-stage classifier on extended feature matrix
        self.clf2_ = clone(self.clf)
        self.clf2_.fit(X_ext, y)
        
        # fill in further relevant attributes
        self.classes_ = self.clf2_.classes_
                
        # return the fitted classifier
        return self

    def predict(self, X):
        """ Predict class labels for samples in X.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            The feature matrix for which we want to get the predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        # check that fit() has been called
        check_is_fitted(self, ('clf1_', 'clf2_', 'offsets_'))
        
        # check that input is a suitable array
        X = check_array(X, accept_sparse='csr')
        
        # apply first-stage classifier
        X_prob = self.clf1_.predict_proba(X)
        
        # shift to obtain context tag probabilities
        Xs = [ np.roll(X_prob, offset, axis=0) for offset in self.offsets_ ]
        
        # apply second-stage classifier to extended feature matrix
        X_ext = sp.sparse.hstack([X] + Xs, format='csr')
        return self.clf2_.predict(X_ext)        

    ## Implicit methods
    # - clf.score(X, y) is automatically provided by the ClassifierMixin
