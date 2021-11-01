"""
Loss functions.
"""

import numpy as np

from stats_learn.util import check_data_shape


def _check_inputs(h, y, shape):
    """
    Check loss function input shapes.

    Parameters
    ----------
    h : array_like
        Decisions.
    y : array_like
        Target values.
    shape : tuple
        Shape of data tensors.

    Returns
    -------
    np.ndarray
    np.ndarray
    tuple

    """
    h, set_shape_est = check_data_shape(h, shape)
    y, set_shape = check_data_shape(y, shape)
    if set_shape_est != set_shape:
        raise ValueError("Input must have same shape.")
    return h, y, set_shape


def loss_se(y_est, y, shape=()):
    y_est, y, set_shape = _check_inputs(y_est, y, shape)
    return ((y_est - y) ** 2).reshape(*set_shape, -1).sum(-1)


def loss_01(y_hyp, y, shape=()):
    y_hyp, y, set_shape = _check_inputs(y_hyp, y, shape)
    return 1 - (y_hyp == y).reshape(*set_shape, -1).all(axis=-1)
