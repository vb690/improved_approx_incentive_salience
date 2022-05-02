import numpy as np

import tensorflow.keras.backend as K


def mae_np(y_true, y_pred, axis=None):
    """Short summary.

    Args:
        y_true (type): Description of parameter `y_true`.
        y_pred (type): Description of parameter `y_pred`.
        axis (type): Description of parameter `axis`.

    Returns:
        type: Description of returned object.

    """
    absolute_error = np.abs(y_true - y_pred)
    mean_absolute_eror = np.nanmean(absolute_error, axis=axis)
    return mean_absolute_eror


def r2_np(y_true, y_pred, axis=None):
    """Short summary.

    Args:
        y_true (type): Description of parameter `y_true`.
        y_pred (type): Description of parameter `y_pred`.
        axis (type): Description of parameter `axis`.

    Returns:
        type: Description of returned object.

    """
    ssres = np.sum((y_true - y_pred) ** 2, axis=axis)
    sstot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=axis)
    r2 = 1 - (ssres / (sstot + 1e-10))
    return r2


def smape_np(y_true, y_pred, axis=None):
    """Short summary.

    Args:
        y_true (type): Description of parameter `y_true`.
        y_pred (type): Description of parameter `y_pred`.
        axis (type): Description of parameter `axis`.

    Returns:
        type: Description of returned object.

    """
    nominator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) + 1e-07
    division = np.nanmean(nominator / denominator, axis=axis)
    return division


def smape_k(y_true, y_pred):
    """Short summary.

    Args:
        y_true (type): Description of parameter `y_true`.
        y_pred (type): Description of parameter `y_pred`.

    Returns:
        type: Description of returned object.

    """
    y_true = K.cast(y_true, "float32")
    nominator = K.abs(y_true - y_pred)
    denominator = K.abs(y_true) + K.abs(y_pred) + K.epsilon()
    division = K.mean(nominator / denominator, axis=-1)
    return division
