"""Core data pre-processing utilities."""
import numpy as np
import pandas as pd


def labels_to_one_hot_code(labels, n=None):
    """Transform labels to one-hot-code."""
    if n is None:
        n = len(set(labels))
    eye_n = np.eye(n)
    return np.array([eye_n[int(label)] for label in labels])


def labels_to_one_hot_code_using_dict(labels, labels_dict):
    """Transform labels to one-hot-code."""
    an_eye = np.eye(len(labels_dict))
    return np.array([an_eye[labels_dict[label]] for label in labels])


def enforce_pandas_dataframe_on_second_argument(function):
    """Decorate to enforce pandas DataFrame argument as input."""

    def _wrapper(first, second):
        return function(first, pd.DataFrame(second))

    return _wrapper
