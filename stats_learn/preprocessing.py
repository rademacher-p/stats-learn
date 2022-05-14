"""Functions for preprocessing observations before training/prediction."""

import math

import numpy as np

from stats_learn.util import check_data_shape


def make_discretizer(vals):  # TODO: use sklearn.preprocessing.KBinsDiscretizer?
    """
    Create a rounding discretization function.

    Parameters
    ----------
    vals : array_like
        Values to which to round.

    Returns
    -------
    function
        The rounding function.

    """
    vals = np.array(vals)

    shape = vals.shape[1:]
    size = math.prod(shape)

    if shape == ():
        vals = np.sort(vals)[
            ::-1
        ]  # trick to break ties towards higher values, for subsets closed on the lower end
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
        n += np.all([i > 0, i < size - 1], axis=0)
    p[idx] = 2**n
    return p


def make_clipper(lims):
    """
    Create a function that clips inputs into a closed box space.

    Parameters
    ----------
    lims : array_like
        Array of lower and upper bounds to clip inputs to.

    Returns
    -------
    function
        Clipping function.

    """
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
