from functools import wraps
from numbers import Integral
from datetime import datetime

import numpy as np


def get_now():
    return datetime.now().replace(microsecond=0).isoformat().replace(':', '_')


class RandomGeneratorMixin:
    def __init__(self, rng=None):
        self.rng = rng

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, val):
        self._rng = self.make_rng(val)

    def _get_rng(self, rng=None):
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
        rng : int or RandomState or Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Generator

        """
        if rng is None:
            return np.random.default_rng()
        elif isinstance(rng, (Integral, np.integer)):
            return np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator) or isinstance(rng, np.random.RandomState):
            return rng
        else:
            raise TypeError("Input must be None, int, or a valid NumPy random number generator.")


def check_data_shape(x, shape=()):
    x = np.array(x)

    idx = x.ndim - len(shape)
    if x.shape[idx:] == shape:
        set_shape = x.shape[:idx]
    else:
        raise TypeError(f"Trailing dimensions of 'x.shape' must be equal to {shape}.")

    # if data_shape == ():      # TODO
    #     set_shape = x.shape
    # # elif x.shape == shape:
    # #     set_shape = ()
    # elif x.shape[-len(data_shape):] == data_shape:
    #     set_shape = x.shape[:-len(data_shape)]
    # else:
    #     raise TypeError("Trailing dimensions of 'x.shape' must be equal to 'data_shape_x'.")

    return x, set_shape


# def check_set_shape(x, set_shape=()):
#     x = np.array(x)
#
#     # if set_shape == ():
#     #     shape = x.shape
#     # elif x.shape == set_shape:
#     #     shape = ()
#     if x.shape[:len(set_shape)] == set_shape:
#         data_shape = x.shape[len(set_shape):]
#     else:
#         raise TypeError("Leading dimensions of 'x.shape' must be equal to 'set_shape'.")
#
#     return x, data_shape


def check_valid_pmf(p, shape=None, full_support=False, tol=1e-9):
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

    if not np.allclose(p.reshape(*set_shape, -1).sum(-1), 1., rtol=tol):
        raise ValueError("The input 'p' must lie within the normal simplex, but p.sum() = %s." % p.sum())

    if shape is None:
        return p
    else:
        return p, set_shape


def vectorize_func(func, shape):
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


