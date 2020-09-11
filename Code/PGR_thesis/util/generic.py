from numbers import Integral
import numpy as np


def check_rng(rng=None):
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


def check_data_shape(x, data_shape=()):
    x = np.asarray(x)

    if data_shape == ():
        set_shape = x.shape
    elif x.shape == data_shape:
        set_shape = ()
    elif x.shape[-len(data_shape):] == data_shape:
        set_shape = x.shape[:-len(data_shape)]
    else:
        raise TypeError("Trailing dimensions of 'x.shape' must be equal to 'data_shape_x'.")

    return x, set_shape


def check_set_shape(x, set_shape=()):
    x = np.asarray(x)

    if set_shape == ():
        data_shape = x.shape
    elif x.shape == set_shape:
        data_shape = ()
    elif x.shape[:len(set_shape)] == set_shape:
        data_shape = x.shape[len(set_shape):]
    else:
        raise TypeError("Leading dimensions of 'x.shape' must be equal to 'set_shape'.")

    return x, data_shape


def check_valid_pmf(p, data_shape=None, full_support=False):
    if data_shape is None:
        p = np.asarray(p)
        set_shape = ()
    else:
        p, set_shape = check_data_shape(p, data_shape)

    if full_support:
        if np.min(p) <= 0:
            raise ValueError("Each entry in 'p' must be positive.")
    else:
        if np.min(p) < 0:
            raise ValueError("Each entry in 'p' must be non-negative.")

    if (np.abs(p.reshape(set_shape + (-1,)).sum(-1) - 1.0) > 1e-9).any():
        raise ValueError("The input 'p' must lie within the normal simplex, but p.sum() = %s." % p.sum())

    return p


def vectorize_func(func, data_shape):
    def func_vec(x):
        x, set_shape = check_data_shape(x, data_shape)

        _out = []
        for x_i in x.reshape((-1,) + data_shape):
            _out.append(func(x_i))
        _out = np.array(_out)

        if len(_out) == 1:      # FIXME: new, check.
            return _out[0]
        else:
            out_shape = _out.shape[1:]
            return _out.reshape(set_shape + out_shape)

    return func_vec


def vectorize_func_dec(data_shape):     # TODO: use?
    def wrapper(func):
        def func_vec(x):
            x, set_shape = check_data_shape(x, data_shape)

            _out = []
            for x_i in x.reshape((-1,) + data_shape):
                _out.append(func(x_i))
            _out = np.asarray(_out)

            if len(_out) == 1:      # FIXME: new, check.
                return _out[0]
            else:
                out_shape = _out.shape[1:]
                return _out.reshape(set_shape + out_shape)

        return func_vec
    return wrapper


def empirical_pmf(d, supp, data_shape):
    """Generates the empirical PMF for a data set."""

    supp, supp_shape = check_data_shape(supp, data_shape)
    n_supp = int(np.prod(supp_shape))
    supp_flat = supp.reshape(n_supp, -1)

    if d.size == 0:
        return np.zeros(supp_shape)

    d, _set_shape = check_data_shape(d, data_shape)
    n = int(np.prod(_set_shape))
    d_flat = d.reshape(n, -1)

    dist = np.zeros(n_supp)
    for d_i in d_flat:
        eq_supp = np.all(d_i.flatten() == supp_flat, axis=-1)
        if eq_supp.sum() != 1:
            raise ValueError("Data must be in the support.")

        dist[eq_supp] += 1

    return dist.reshape(supp_shape) / n
