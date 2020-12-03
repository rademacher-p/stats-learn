import math

import numpy as np
# from matplotlib import pyplot as plt
from scipy.special import binom


# def get_axes_xy(ax=None, shape=None):   # TODO: delete?
#     if ax is None:
#         if shape == ():
#             _, ax = plt.subplots()
#             ax.set(xlabel='$x$', ylabel='$y$')
#         elif shape == (2,):
#             _, ax = plt.subplots(subplot_kw={'projection': '3d'})
#             ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$y$')
#         else:
#             return None
#
#         ax.grid(True)
#         return ax
#     else:
#         return ax


def simplex_grid(n, shape, hull_mask=None):
    """
    Generate a uniform grid over a simplex.
    """

    if type(n) is not int or n < 1:
        raise TypeError("Input 'n' must be a positive integer")

    if type(shape) is not tuple:
        raise TypeError("Input 'shape' must be a tuple of integers.")
    elif not all([isinstance(x, int) for x in shape]):
        raise TypeError("Elements of 'shape' must be integers.")

    d = math.prod(shape)

    if hull_mask is None:
        hull_mask = np.broadcast_to(False, (d,))
    # elif hull_mask == 'all':
    #     hull_mask = np.broadcast_to(True, math.prod(shape))
    else:
        hull_mask = np.asarray(hull_mask)
        if hull_mask.shape != shape:
            raise TypeError("Input 'hull_mask' must have same shape.")
        elif not all([isinstance(x, np.bool_) for x in hull_mask.flatten()]):
            raise TypeError("Elements of 'hull_mask' must be boolean.")
        hull_mask = hull_mask.flatten()

    if n < sum(hull_mask.flatten()):
        raise ValueError("Input 'n' must meet or exceed the number of True values in 'hull_mask'.")

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


# g = simplex_grid(10, shape=(3,))
# print(g.shape)
# print(binom(12, 2))


def box_grid(lims, n=100, endpoint=False):
    lims = np.array(lims)

    if endpoint:
        n += 1

    if lims.shape == (2,):
        return np.linspace(*lims, n, endpoint=endpoint)
    elif lims.ndim == 2 and lims.shape[-1] == 2:
        x_dim = [np.linspace(*lims_i, n, endpoint=endpoint) for lims_i in lims]
        # return np.stack(np.meshgrid(*x_dim), axis=-1)
        return mesh_grid(*x_dim)
    else:
        raise ValueError("Shape must be (2,) or (*, 2)")

    # if not (lims[..., 0] <= lims[..., 1]).all():
    #     raise ValueError("Upper values must meet or exceed lower values.")


def mesh_grid(*args):
    # return np.stack(np.meshgrid(*args[::-1])[::-1], axis=-1)
    return np.stack(np.meshgrid(*args, indexing='ij'), axis=-1)
