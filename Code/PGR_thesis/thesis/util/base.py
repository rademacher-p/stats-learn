from functools import wraps
from itertools import groupby
from numbers import Integral

import numpy as np

DELTA = 1e250  # large value approximating the value of the Dirac delta function at zero


class RandomGeneratorMixin:
    def __init__(self, rng=None):
        self.rng = rng

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, val):
        self._rng = self.check_rng(val)

    def _get_rng(self, rng=None):
        if rng is None:
            return self._rng
        else:
            return self.check_rng(rng)

    @staticmethod
    def check_rng(rng):
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

    # if (np.abs(p.reshape(*set_shape, -1).sum(-1) - 1.0) > tol).any():
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


def vectorize_func_dec(shape):  # TODO: use?
    def wrapper(func):
        @wraps(func)
        def func_vec(x):
            x, set_shape = check_data_shape(x, shape)

            _out = []
            for x_i in x.reshape((-1,) + shape):
                _out.append(func(x_i))
            _out = np.asarray(_out)

            # if len(_out) == 1:
            #     return _out[0]
            # else:
            return _out.reshape(set_shape + _out.shape[1:])

        return func_vec

    return wrapper

# def vectorize_first_arg(func):
#     @wraps(func)
#     def func_wrap(*args, **kwargs):
#         if isinstance(args[0], Iterable):
#             return list(func(arg, *args[1:], **kwargs) for arg in args[0])
#         else:
#             return func(*args, **kwargs)
#
#     return func_wrap


# def empirical_pmf(d, supp, shape):
#     """Generates the empirical PMF for a data set."""
#
#     supp, supp_shape = check_data_shape(supp, shape)
#     n_supp = math.prod(supp_shape)
#     supp_flat = supp.reshape(n_supp, -1)
#
#     if d.size == 0:
#         return np.zeros(supp_shape)
#
#     d, _set_shape = check_data_shape(d, shape)
#     n = math.prod(_set_shape)
#     d_flat = d.reshape(n, -1)
#
#     dist = np.zeros(n_supp)
#     for d_i in d_flat:
#         eq_supp = np.all(d_i.flatten() == supp_flat, axis=-1)
#         if eq_supp.sum() != 1:
#             raise ValueError("Data must be in the support.")
#
#         dist[eq_supp] += 1
#
#     return dist.reshape(supp_shape) / n


def all_equal(iterable):
    """Returns True if all the elements are equal to each other"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)
