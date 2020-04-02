import numpy as np
# from scipy.special import binom


def simplex_grid(n=0, d=2):
    """
    Generate a uniform grid over a simplex.

    :param n: the number of points per dimension, minus one
    :param d: dimension of the simplex, plus one
    :return: (m,d) array, where m is the total number of points
    """

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

    return g / n


# if __name__ == '__main__':
#     print(simplex_grid(2, 3))
