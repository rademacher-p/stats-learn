import numpy as np
# from scipy.special import binom

#%% Mathematics

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


#%% Plotting

def simplex_grid(n=1, shape=(1,), hull_mask=None):
    """
    Generate a uniform grid over a simplex.

    :param n: the number of points per dimension, minus one
    :param shape: shape of the simplex samples
    :param hull_mask: boolean array dictating which simplex edge boundaries to exclude
    :return: (m,)+shape array, where m is the total number of points
    """

    if type(n) is not int or n < 1:
        raise TypeError("Input 'n' must be a positive integer")

    if type(shape) is not tuple:
        raise TypeError("Input 'shape' must be a tuple of integers.")
    elif not all([isinstance(x, int) for x in shape]):
        raise TypeError("Elements of 'shape' must be integers.")

    if hull_mask is None:
        hull_mask = np.broadcast_to(False, np.prod(shape))
    # elif hull_mask == 'all':
    #     hull_mask = np.broadcast_to(True, np.prod(shape))
    else:
        hull_mask = np.asarray(hull_mask)
        if hull_mask.shape != shape:
            raise TypeError("Input 'hull_mask' must have same shape.")
        elif not all([isinstance(x, np.bool_) for x in hull_mask.flatten()]):
            raise TypeError("Elements of 'hull_mask' must be boolean.")
        hull_mask = hull_mask.flatten()

    d = np.prod(shape)

    if d == 1:
        return np.array(1).reshape(shape)

    s = 1 if hull_mask[0] else 0
    e = 0 if (d == 2 and hull_mask[1]) else 1
    g = np.arange(s, n + e)[:, np.newaxis]
    # g = np.arange(n + 1)[:, np.newaxis]

    for i in range(1, d-1):
        s = 1 if hull_mask[i] else 0
        e = 0 if (i == d-2 and hull_mask[i+1]) else 1

        g_new = []
        for v in g:
            # for k in np.arange(n+1 - g_i.sum()):
            for k in np.arange(s, n + e - v.sum()):
                g_new.append(np.append(v, k))
        g = np.array(g_new)

    g = np.hstack((g, n - g.sum(axis=1)[:, np.newaxis]))

    return g.reshape((-1,) + shape) / n


if __name__ == '__main__':
    q = simplex_grid(3, (4,), [False, False, True, False])
    print(q*3)

