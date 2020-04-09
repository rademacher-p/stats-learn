import numpy as np


def outer_gen(x, y):
    x, y = np.asarray(x), np.asarray(y)
    x_dims, y_dims = x.ndim, y.ndim
    for i in range(y_dims):
        x = x[:, np.newaxis]
    for i in range(x_dims):
        y = y[np.newaxis, :]
    return x*y


def diag_gen(x):
    x = np.asarray(x)
    out = np.zeros(2*x.shape)
    i = np.unravel_index(range(x.size), x.shape)
    out[2*i] = x.flatten()
    return out
