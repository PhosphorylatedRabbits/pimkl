"""Split data into training and test."""
import numpy as np


def _get_learning_data_indices(samples, labels=None, max_per_class=30):
    """In case labels are None only one class named 'class' is considered."""
    permuted = np.random.permutation(range(samples))
    if labels is None:
        labels = np.repeat('class', samples)
    counts = {label: 0 for label in set(labels)}
    train = []
    test = []
    for index in permuted:
        label = labels[index]
        if counts[label] < max_per_class:
            counts[label] += 1
            train.append(index)
        else:
            test.append(index)
    return train, test


def get_learning_data_indices_fraction(X, fraction=0.5):
    """Return data in dict mode splitted using a fraction."""
    # get a key
    if isinstance(X, dict):
        keys = list(X.keys())
        number_of_samples = X[keys[0]].shape[0]
    else:
        number_of_samples = X.shape[0]
    sample_indices = np.arange(number_of_samples)
    number_of_training_samples = int(np.floor(fraction * number_of_samples))
    train = list(np.random.choice(sample_indices, number_of_training_samples))
    test = list(set(sample_indices) - set(train))
    return train, test


def get_learning_data_in_dict_mode(
    X, labels=None, data_types=None, max_per_class=30
):
    """Return splitted test and training data for multiple data types."""
    if data_types is None:
        data_types = list(X.keys())
    number_of_samples = X[data_types[0]].shape[0]

    train, test = _get_learning_data_indices(
        number_of_samples, labels, max_per_class=max_per_class
    )
    X_train, X_test = {}, {}
    for data_type in data_types:
        X_train[data_type], X_test[data_type] = \
            X[data_type][train], X[data_type][test]
    if labels is None:
        return X_train, X_test
    else:
        y_train, y_test = labels[train], labels[test]
        return X_train, y_train, X_test, y_test


def get_learning_data_in_dict_mode_fraction(
    X, labels=None, data_types=None, fraction=0.5
):
    """Return splitted test and training data for multiple data types."""
    if data_types is None:
        data_types = list(X.keys())

    train, test = get_learning_data_indices_fraction(X, fraction=fraction)
    X_train, X_test = {}, {}
    for data_type in data_types:
        X_train[data_type], X_test[data_type] = \
            X[data_type][train], X[data_type][test]
    if labels is None:
        return X_train, X_test
    else:
        y_train, y_test = labels[train], labels[test]
        return X_train, y_train, X_test, y_test


def get_learning_data(X, labels=None, max_per_class=30):
    """Return splitted test and training data for single data type."""
    number_of_samples = X.shape[0]
    train, test = _get_learning_data_indices(
        number_of_samples, labels, max_per_class
    )
    if labels is None:
        return X[train], X[test]
    else:
        return X[train], labels[train], X[test], labels[test]
