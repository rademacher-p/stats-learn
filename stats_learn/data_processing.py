import math

import numpy as np

from stats_learn.util import check_data_shape
# from stats_learn.util.plotting import box_grid


def make_discretizer(vals):  # TODO: use sklearn.preprocessing.KBinsDiscretizer?
    """Create a rounding discretization function."""
    vals = np.array(vals)

    shape = vals.shape[1:]
    size = math.prod(shape)

    if shape == ():
        vals = np.sort(vals)[::-1]  # trick to break ties towards higher values, for subsets closed on the lower end
    vals_flat = vals.reshape(-1, size)

    def discretizer(x):
        x, set_shape = check_data_shape(x, shape)
        x = x.reshape(-1, size)

        delta = x - vals_flat[:, np.newaxis]
        idx = (np.linalg.norm(delta, axis=-1)).argmin(axis=0)

        return vals[idx].reshape(set_shape + shape)

    return discretizer


def prob_disc(shape):
    """Create (unnormalized) probability array for a discretization grid. Lower edge/corner probabilities."""

    p = np.ones(shape)
    idx = np.nonzero(p)
    n = np.zeros(p.size)
    for i, size in zip(idx, shape):
        n += np.all([i > 0, i < size-1], axis=0)
    p[idx] = 2**n
    return p


def make_clipper(lims):
    """Create a function that clips inputs into a closed box space."""
    lims = np.array(lims)

    low, high = lims[..., 0], lims[..., 1]
    if lims.shape[-1] != 2:
        raise ValueError("Trailing shape must be (2,)")
    elif not np.all(low <= high):
        raise ValueError("Upper values must meet or exceed lower values.")

    def clipper(x):
        x = np.where(x < low, low, x)
        x = np.where(x > high, high, x)
        return x

    return clipper


# def main():
#     # test discretizer
#     x = np.random.default_rng().random(10)
#     print(x)
#
#     vals = np.linspace(0, 1, 11)
#     func_ = make_discretizer(vals)
#     x_d = func_(x)
#     print(x_d)
#
#     x = np.random.default_rng().random((10, 2))
#     print(x)
#
#     vals = box_grid([[0, 1], [0, 1]], 11, True).reshape(-1, 2)
#     func_ = make_discretizer(vals)
#     x_d = func_(x)
#     print(x_d)
#
#     # test clipper
#     # lims = np.array((0, 1))
#     lims = np.array([(0, 1), (0, 1)])
#     clipper = make_clipper(lims)
#
#     x = np.random.default_rng().uniform(-1, 2, (10, *lims.shape[:-1]))
#     print(x)
#
#     x_c = clipper(x)
#     print(x_c)
#
#
# if __name__ == '__main__':
#     main()
