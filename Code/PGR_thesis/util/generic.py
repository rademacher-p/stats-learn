import numpy as np
# from scipy.special import binom


def check_data_shape(x, data_shape):
    x = np.asarray(x)

    if x.shape == data_shape:
        set_shape = ()
    elif data_shape == ():
        set_shape = x.shape
    elif x.shape[-len(data_shape):] == data_shape:
        set_shape = x.shape[:-len(data_shape)]
    else:
        raise TypeError("Trailing dimensions of 'x.shape' must be equal to 'data_shape'.")

    return x, set_shape


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


def vectorize_x_func(func, data_shape):
    def func_vec(x):
        x, set_shape = check_data_shape(x, data_shape)

        _out = []
        for x_i in x.reshape((-1,) + data_shape):
            _out.append(func(x_i))
        _out = np.asarray(_out)

        return _out.reshape(set_shape + _out.shape[1:])

    return func_vec


def empirical_pmf(d, supp, data_shape):
    """Generates the empirical PMF for a data set."""

    d, _set_shape = check_data_shape(d, data_shape)
    n = int(np.prod(_set_shape))
    d_flat = d.reshape(n, -1)

    supp, supp_shape = check_data_shape(supp, data_shape)
    n_supp = int(np.prod(supp_shape))
    supp_flat = supp.reshape(n_supp, -1)

    dist = np.zeros(n_supp)
    for d_i in d_flat:
        eq_supp = np.all(d_i.flatten() == supp_flat, axis=-1)
        if eq_supp.sum() != 1:
            raise ValueError("Data must be in the support.")

        dist[eq_supp] += 1

    return dist.reshape(supp_shape) / n

