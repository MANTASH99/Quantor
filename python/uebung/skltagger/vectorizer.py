"""
SKLTagger: Feature extraction
"""
import numpy as np
import scipy as sp
import pandas as pd
import re

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_scalar, check_is_fitted, check_consistent_length
from sklearn.utils.multiclass import unique_labels

from sklearn.feature_extraction.text import CountVectorizer


class TaggerFeatures(TransformerMixin, BaseEstimator):
    """ Extract feature matrix for each token in a sequence of sentences.

    Input must be provided as a Pandas data frame with one token per line,
    a column ``word`` specifying the word form and a column ``sent`` specifying 
    the sentence number (used to detect sentence boundaries). Any other 
    columns will be ignored.
    
    **TODO:** Write full documentation for this class and its methods, following
    the NumPy docstring standard:
    https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
    
    Parameters
    ----------
    word_min_f : inf, default=1
        Frequency threshold for the one-hot encoding of the word form itself.
    word_lc : bool, default=False
        Fold word form features to lowercase.
    suffix_len : int, default=5
        Maximal number of characters for suffix features (must be >= 2).
    prefix_len : int, default=3
        Maximal number of characters for prefix features (must be >= 2).
    affix_min_f : int, default=1
        Frequency threshold for suffix and prefix features.
    affix_lc = bool, default=True
        Fold suffix and prefix features to lowercase.
    shape_features : bool, default=True
        Include further binary features describing the orthographic shape of words 
        (e.g. whether they start with an uppercase letter).
    left_context : int, default=1
        Number of tokens before the current token to use as context.
    right_context : int, default=1
        Number of tokens after the current token to use as context.
    dtype : dtype, default=np.int64
        Type of matrix returned by fit_transform() or transform().

    Attributes
    ----------
    word_vectorizer_ : object
        Vectorizer for one-hot encoding of word forms.

    **TODO** 
    """
    def __init__(self, word_min_f=1, word_lc=False,
                       suffix_len=5, prefix_len=3, affix_min_f=1, affix_lc=True,
                       shape_features=True,
                       left_context=1, right_context=1,
                       dtype=np.int64):
        self.word_min_f = word_min_f
        self.word_lc = word_lc
        self.suffix_len = suffix_len
        self.prefix_len = prefix_len
        self.affix_min_f = affix_min_f
        self.affix_lc = affix_lc
        self.shape_features = shape_features
        self.left_context = left_context
        self.right_context = right_context
        self.dtype = dtype


    ## internal helper methods
    def get_prefix_suffix(self, word):
        l = len(word)
        res = []
        for k in range(2, min(self.suffix_len + 1, l)):
            res.append("-" + word[-k:])
        for k in range(2, min(self.prefix_len + 1, l)):
            res.append(word[:k] + "-")
        return(res)
    
    def get_shape_features(self, words, get_names=False):
        if get_names:
            return "upper allcaps digits noalpha noalnum trunc long".split()
        res = pd.DataFrame({
            'upper': words.str.match(r'[A-ZÄÖÜ]'),
            'allcaps': words.str.fullmatch(r'[A-ZÄÖÜ]+'),
            'digits': words.str.fullmatch(r'-?[0-9][0-9.,]*'),
            'noalpha': ~ words.str.contains(r'[a-zäöü]', flags=re.IGNORECASE),
            'noalnum': ~ words.str.contains(r'[0-9a-zäöü]', flags=re.IGNORECASE),
            'trunc': words.str.endswith('-'),
            'long': words.str.len() >= 15,
        })
        return res.to_numpy(dtype=self.dtype)
    
    # shifting by +2 returns L2 context word, shifting by -1 returns R1 context word
    def shift_tokens(self, tokens, by=0, padding=''):
        shifted_word = tokens.word.shift(by)
        shifted_sent = tokens.sent.shift(by, fill_value=-1)
        shifted_word.where(tokens.sent == shifted_sent, padding, inplace=True)
        return shifted_word
    
    def check_X_data_frame(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a Pandas data frame")
        if not {'word', 'sent'}.issubset(X.columns):
            raise ValueError("input data frame must have columns 'word' and 'sent'")
    
    def none_tokenizer(self, x):
        return (x,)
    
    ## public API
    def fit(self, tokens, y=None):
        """ Learn vocabulary dictionaries for word form, suffix and prefix features.

        Parameters
        ----------
        tokens : Pandas data frame with columns ``word`` and ``sent``
            Training data as a data frame with one token per line.
        y : None
            This parameter is ignored.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        # check that input is well-formed
        self.check_X_data_frame(tokens)
        
        # validate all parameter settings (must be done here, not in __init__)
        check_scalar(self.word_min_f, 'word_min_f', target_type=int, min_val=1)
        check_scalar(self.suffix_len, 'suffix_len', target_type=int, min_val=2)
        check_scalar(self.prefix_len, 'prefix_len', target_type=int, min_val=2)
        check_scalar(self.affix_min_f, 'affix_min_f', target_type=int, min_val=1)
        check_scalar(self.left_context, 'left_context', target_type=int, min_val=0, max_val=9)
        check_scalar(self.right_context, 'right_context', target_type=int, min_val=0, max_val=9)
        # TODO: check other parameters (bool, data type)

        # train suitable one-hot encoders for word forms and prefix/suffix features
        # NB: we use CountVectorizers for both, which don't represent OOVs explicitly
        self.word_vectorizer_ = CountVectorizer(tokenizer=self.none_tokenizer, token_pattern=None,
                                                lowercase=self.word_lc, min_df=self.word_min_f, dtype=self.dtype)
        self.word_vectorizer_.fit(tokens.word)

        self.affix_vectorizer_ = CountVectorizer(tokenizer=self.get_prefix_suffix, token_pattern=None,
                                                 lowercase=self.affix_lc, min_df=self.affix_min_f, dtype=self.dtype)
        self.affix_vectorizer_.fit(tokens.word)
        
        # return the transformer
        return self

    def transform(self, tokens):
        """ Extract feature vectors for the input tokens.

        Parameters
        ----------
        tokens : Pandas data frame with columns ``word`` and ``sent``
            Input data as a data frame with one token per line.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Sparse matrix of feature vectors extracted from the input data.
        """
        # check that fit() has been called
        check_is_fitted(self, ('word_vectorizer_', 'affix_vectorizer_'))
        
        # validate input, and hope that no-one has messed with parameter settings since calling fit()
        self.check_X_data_frame(tokens)
        
        Xs = [] # list of partial feature matrices
        
        # features for the word to be tagged
        Xs.append(self.word_vectorizer_.transform(tokens.word))
        Xs.append(self.affix_vectorizer_.transform(tokens.word))
        if self.shape_features:
            Xs.append(self.get_shape_features(tokens.word))

        # left and right context features (offsets -1 = R1, +2, = L2)
        offsets = [x for x in range(-self.right_context, self.left_context+1) if x != 42] 
        for i in offsets:
            shifted_words = self.shift_tokens(tokens, i)
            Xs.append(self.word_vectorizer_.transform(shifted_words))
            Xs.append(self.affix_vectorizer_.transform(shifted_words))
            if self.shape_features:
                Xs.append(self.get_shape_features(shifted_words))
        
        return sp.sparse.hstack(Xs, format='csr')

    
    ## optional API methods
    def get_feature_names_out(self):
        """ Return list of feature names for a fitted vectorizer.
        
        Returns
        -------
        features : ndarray of str objects
            List of feature names.
        """
        check_is_fitted(self, ('word_vectorizer_', 'affix_vectorizer_'))

        fnames = self.word_vectorizer_.get_feature_names_out().tolist()
        fnames += self.affix_vectorizer_.get_feature_names_out().tolist()
        if self.shape_features:
            fnames += self.get_shape_features(None, get_names=True)
        
        features = fnames.copy()
        for i in range(self.right_context, 0, -1):
            features += [f'R{i}_{x}' for x in fnames]
        for i in range(1, self.left_context + 1):
            features += [f'L{i}_{x}' for x in fnames]
        
        return np.asarray(features, dtype=object)
