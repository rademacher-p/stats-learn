import math

import numpy as np

from stats_learn.util.base import check_data_shape
from stats_learn.util.math import prob_disc
from stats_learn.util.plotting import box_grid


def discretizer(vals):  # TODO: use sklearn.preprocessing.KBinsDiscretizer?
    """Create a rounding discretizer."""
    vals = np.array(vals)

    shape = vals.shape[1:]
    size = math.prod(shape)

    if shape == ():
        vals = np.sort(vals)[::-1]  # trick to break ties towards higher values, for subsets closed on the lower end
    vals_flat = vals.reshape(-1, size)

    def func(x):
        x, set_shape = check_data_shape(x, shape)
        x = x.reshape(-1, size)

        delta = x - vals_flat[:, np.newaxis]
        idx = (np.linalg.norm(delta, axis=-1)).argmin(axis=0)

        return vals[idx].reshape(set_shape + shape)

    return func


def main():
    x = np.random.default_rng().random(10)
    print(x)

    vals = np.linspace(0, 1, 11, endpoint=True)
    func_ = discretizer(vals)
    x_d = func_(x)
    print(x_d)

    x = np.random.default_rng().random((10, 2))
    print(x)

    vals = box_grid([[0, 1], [0, 1]], 11, True).reshape(-1, 2)
    func_ = discretizer(vals)
    x_d = func_(x)
    print(x_d)

    print(prob_disc((4, 3)))


if __name__ == '__main__':
    main()
