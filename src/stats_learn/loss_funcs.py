"""Loss functions for regression and classification."""

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
    numpy.ndarray
        Decisions.
    numpy.ndarray
        Target values.
    tuple
        Shape of data set (such that `value + shape = h.shape`)

    """
    h, y = np.broadcast_arrays(h, y)
    h, set_shape_est = check_data_shape(h, shape)
    y, set_shape = check_data_shape(y, shape)
    if set_shape_est != set_shape:
        raise ValueError("Input must have same shape.")
    return h, y, set_shape


def make_loss_lp(p: float):
    """Make loss function for generic powers of the absolute squared-difference.

    Parameters
    ----------
    p : float
        The power.

    """

    def loss_lp(y_est, y, shape=()):
        y_est, y, set_shape = _check_inputs(y_est, y, shape)
        return (np.abs(y_est - y) ** p).reshape(*set_shape, -1).sum(-1)

    return loss_lp


loss_ae = make_loss_lp(1)
loss_se = make_loss_lp(2)

# def loss_se(y_est, y, shape=()):
#     """
#     Squared-error loss function.

#     Parameters
#     ----------
#     y_est : array_like
#         The predictors.
#     y : array_like
#         The targets.
#     shape : tuple, optional
#         Shape of data tensors.

#     Returns
#     -------
#     numpy.ndarray
#         The squared-error losses.

#     """
#     y_est, y, set_shape = _check_inputs(y_est, y, shape)
#     return ((y_est - y) ** 2).reshape(*set_shape, -1).sum(-1)


def loss_01(y_hyp, y, shape=()):
    """
    Zero-one loss function.

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
    numpy.ndarray
        The zero-one losses.

    """
    y_hyp, y, set_shape = _check_inputs(y_hyp, y, shape)
    return 1 - (y_hyp == y).reshape(*set_shape, -1).all(axis=-1)
