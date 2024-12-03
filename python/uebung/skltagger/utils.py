"""
SKLTagger: Utility functions
"""

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from .vectorizer import TaggerFeatures

def sentences2dataframe(sentence_list):
    """ Transform a list of tokenized sentences into a Pandas DataFrame.
    
    Parameters
    ----------
    sentence_list : iterable over list of str
        Pre-tokenized sentences. If there is only a single sentence, it must be passed as a list of length 1.

    Returns
    -------
    tokens : DataFrame
        Pandas data frame with one token per line in column ``word`` and consecutive sentence numbers in column ``sent``.
    """
    dfs = [ pd.DataFrame({'sent': i+1, 'word': s}) for i, s in enumerate(sentence_list) ]
    return pd.concat(dfs)

def load_model(filename):
    """ Load a pickled SKLTagger model from disk.
    
    Various heuristic checks are carried out to confirm that the disk file contains a valid tagger model.
    
    Parameters
    ----------
    filename : str
        Name / path of the pickle file to be loaded.

    Returns
    -------
    model : object
        The SKLTagger model loaded from disk.
    """
    model = load(filename)
    if not isinstance(model, Pipeline):
        raise ValueError("{} isn't a valid SKLTagger model (must be a Pipeline)".format(filename))
    if not isinstance(model[0], TaggerFeatures):
        raise ValueError("{} isn't a valid SKLTagger model (first component must be a TaggerFeatures vectorizer)".format(filename))
    if not (isinstance(model[-1], BaseEstimator) and isinstance(model[-1], ClassifierMixin)):
        raise ValueError("{} isn't a valid SKLTagger model (first component must be a classifier)".format(filename))
    return model
