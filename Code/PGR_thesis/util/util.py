import numpy as np
# from scipy.special import binom


def _outer_gen_2(x, y):
    x, y = np.asarray(x), np.asarray(y)
    x = x.reshape(x.shape + tuple(np.ones(y.ndim, dtype=int)))  # add singleton dimensions for broadcasting
    return x*y


def outer_gen(*args):
    n_args = len(args)
    if n_args < 2:
        raise TypeError('At least two positional inputs are required.')

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


def simplex_grid(n=1, shape=(2,)):
    """
    Generate a uniform grid over a simplex.

    :param n: the number of points per dimension, minus one
    :param shape: shape of the simplex samples
    :return: (m,)+shape array, where m is the total number of points
    """

    if type(n) is not int or n < 1:
        raise TypeError("Input 'n' must be a positive integer")
    if type(shape) is not tuple:
        raise TypeError("Input 'shape' must be a tuple of integers.")
    elif not all([isinstance(x, int) for x in shape]):
        raise TypeError("Elements of 'shape' must be integers.")

    d = np.prod(shape)

    if d == 1:
        return np.ones(1)

    g = np.arange(n+1)[:, np.newaxis]
    while g.shape[1] < d-1:
        gg = []
        for s in g:
            for k in np.arange(n+1 - s.sum()):
                gg.append(np.append(s, k))
        g = np.array(gg)

    g = np.hstack((g, n - g.sum(axis=1)[:, np.newaxis]))

    # if g.shape[0] != binom(n+d-1, d-1):
    #     raise ValueError('Error: Wrong number of set elements...')

    return g.reshape((-1,) + shape) / n
