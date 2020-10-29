"""
Loss functions.
"""

from thesis.util.generic import check_data_shape


def _check_inputs(y_est, y, data_shape):
    y_est, set_shape_est = check_data_shape(y_est, data_shape)
    y, set_shape = check_data_shape(y, data_shape)
    if set_shape_est != set_shape:
        raise ValueError("Input must have same shape.")
    return y_est, y, set_shape


def loss_se(y_est, y, data_shape=()):
    # y_est, y = np.array(y_est), np.array(y)
    # return ((y_est - y)**2).sum()
    y_est, y, set_shape = _check_inputs(y_est, y, data_shape)
    return ((y_est - y) ** 2).reshape(*set_shape, -1).sum(-1)


def loss_01(y_hyp, y, data_shape=()):
    # return 1 - (y_hyp == y)
    y_hyp, y, set_shape = _check_inputs(y_hyp, y, data_shape)
    return 1 - (y_hyp == y).reshape(*set_shape, -1).all(axis=-1)
