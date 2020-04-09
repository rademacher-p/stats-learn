import numpy as np
# from scipy.special import binom


def simplex_grid(n=0, shape=(2,)):
    """
    Generate a uniform grid over a simplex.

    :param n: the number of points per dimension, minus one
    :param shape: shape of the simplex samples
    :return: (m,)+shape array, where m is the total number of points
    """

    if type(shape) is not tuple:        # TODO: flexibility for int, float inputs?
        raise ValueError('Input "shape" must be a tuple of ints.')

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

    return g.reshape((-1,)+shape) / n
