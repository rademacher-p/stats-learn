import numpy as np


def outer_gen(*args):
    n_args = len(args)
    if n_args < 2:
        return np.asarray(args[0])

    def _outer_gen_2(x, y):
        x, y = np.asarray(x), np.asarray(y)
        x = x.reshape(x.shape + tuple(np.ones(y.ndim, dtype=int)))  # add singleton dimensions for broadcasting
        return x * y

    out = args[0]
    for arg in args[1:]:
        out = _outer_gen_2(out, arg)
    return out


def diag_gen(x):
    x = np.asarray(x)
    out = np.zeros(2*x.shape)
    i = np.unravel_index(range(x.size), x.shape)
    out[2*i] = x.flatten()
    return out


def simplex_round(x):
    x = np.asarray(x)
    if x.min() < 0:
        raise ValueError("Input values must be non-negative.")
    elif x.sum() != 1:
        raise ValueError("Input values must sum to one.")

    out = np.zeros(x.size)
    up = 1
    for i, x_i in enumerate(x.flatten()):
        if x_i < up / 2:
            up -= x_i
        else:
            out[i] = 1
            break

    return out.reshape(x.shape)
