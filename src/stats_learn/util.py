"""Common utilities."""

import sys
from datetime import datetime
from functools import wraps
from numbers import Integral

import numpy as np


def get_now():
    """Get ISO datetime in string format."""
    str_ = datetime.now().isoformat(timespec="seconds")
    if sys.platform.startswith("win32"):
        str_ = str_.replace(":", "_")
    return str_


class RandomGeneratorMixin:
    """
    Provides a RNG property and methods for seeding.

    Parameters
    ----------
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, rng=None):
        self.rng = rng

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, value):
        self._rng = self.make_rng(value)

    def _get_rng(self, rng=None):
        """Return own RNG or make a new one."""
        if rng is None:
            return self._rng
        else:
            return self.make_rng(rng)

    @staticmethod
    def make_rng(rng):
        """
        Return a random number generator.

        Parameters
        ----------
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        np.random.Generator

        """
        if rng is None:
            return np.random.default_rng()
        elif isinstance(rng, Integral | np.integer):
            return np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator | np.random.RandomState):
            return rng
        else:
            raise TypeError(
                "Input must be None, int, or a valid NumPy random number generator."
            )


def check_data_shape(x, shape=()):
    """
    Check that trailing elements of array shape match desired data shape.

    Parameters
    ----------
    x : array_like
        The array.
    shape : tuple, optional
        Shape of data tensors.

    Returns
    -------
    numpy.ndarray
        The data tensor as a NumPy array.
    tuple
        The shape of the data set (i.e., the leading elements of the array shape).

    """
    x = np.array(x)

    idx = x.ndim - len(shape)
    if x.shape[idx:] == shape:
        set_shape = x.shape[:idx]
    else:
        raise TypeError(f"Trailing dimensions of 'x.shape' must be equal to {shape}.")

    return x, set_shape


def check_valid_pmf(p, shape=None, full_support=False, tol=1e-9):
    """
    Check that array is a valid probability mass function.

    Parameters
    ----------
    p : array_like
        The array.
    shape : tuple, optional
        Shape of PMF tensors.
    full_support : bool, optional
        Raises `ValueError` if the PMF is not full-support (i.e. if any elements are
        zero)
    tol : float, optional
        Tolerance for ensuring PMF sums to unity.

    Returns
    -------
    numpy.ndarray
        The PMF as a NumPy array.
    tuple, optional
        The shape of the set of PMFs (i.e., the leading elements of the array shape).

    """
    if shape is None:
        p = np.array(p)
        set_shape = ()
    else:
        p, set_shape = check_data_shape(p, shape)

    if full_support:
        if np.min(p) <= 0:
            raise ValueError("Each entry in 'p' must be positive.")
    else:
        if np.min(p) < 0:
            raise ValueError("Each entry in 'p' must be non-negative.")

    if not np.allclose(p.reshape(*set_shape, -1).sum(-1), 1.0, rtol=tol):
        raise ValueError(
            f"The input 'p' must lie within the normal simplex, but norm = {p.sum()}."
        )

    if shape is None:
        return p
    else:
        return p, set_shape


def vectorize_func(func, shape):
    """
    Vectorize a callable according to specified input shape.

    Parameters
    ----------
    func : callable
        Function to vectorize.
    shape : tuple, optional
        Shape of data tensors.

    Returns
    -------
    function

    """

    @wraps(func)
    def func_vec(x):
        x, set_shape = check_data_shape(x, shape)

        _out = []
        for x_i in x.reshape((-1,) + shape):
            _out.append(func(x_i))
        _out = np.array(_out)
        _out = _out.reshape(set_shape + _out.shape[1:])

        if _out.shape == ():
            return _out.item()
        else:
            return _out

    return func_vec


def make_power_func(i):
    def _func(x):
        # return x**i
        return np.array(x) ** i

        # # y = (x**i).mean()
        # y = (x**i).sum()
        # return np.full(out_shape, y)

        # if out_shape == ():
        #     return y
        # else:
        #     return np.full(out_shape, y)

    return _func
