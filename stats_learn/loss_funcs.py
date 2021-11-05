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
        Decisions.
    np.ndarray
        Target values.
    tuple
        Shape of data set (such that `value + shape = h.shape`)

    """
    h, set_shape_est = check_data_shape(h, shape)
    y, set_shape = check_data_shape(y, shape)
    if set_shape_est != set_shape:
        raise ValueError("Input must have same shape.")
    return h, y, set_shape


def loss_se(y_est, y, shape=()):
    """
    The squared-error loss function.

    Parameters
    ----------
    y_est : array_like
        The predictors.
    y : array_like
        The targets.
    shape : tuple, optional
        Shape of data tensors.

    Returns
    -------
    np.ndarray
        The squared-error losses.

    """
    y_est, y, set_shape = _check_inputs(y_est, y, shape)
    return ((y_est - y) ** 2).reshape(*set_shape, -1).sum(-1)


def loss_01(y_hyp, y, shape=()):
    """
    The zero-one loss function.

    Parameters
    ----------
    y_hyp : array_like
        The hypotheses
    y : array_like
        The targets.
    shape : tuple, optional
        Shape of data tensors.

    Returns
    -------
    np.ndarray
        The zero-one losses.

    """
    y_hyp, y, set_shape = _check_inputs(y_hyp, y, shape)
    return 1 - (y_hyp == y).reshape(*set_shape, -1).all(axis=-1)
