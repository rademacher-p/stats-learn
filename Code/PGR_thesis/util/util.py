import numpy as np
# from scipy.special import binom


#%%

def check_data_shape(x, shape):
    x = np.asarray(x)

    if x.shape == shape:
        set_shape = ()
    elif shape == ():
        set_shape = x.shape
    elif x.shape[-len(shape):] == shape:
        set_shape = x.shape[:-len(shape)]
    else:
        raise TypeError("Trailing dimensions of 'shape' must be equal to the shape of 'x'.")

    return x, set_shape


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


#%% Math operators

def outer_gen(*args):
    n_args = len(args)
    if n_args < 2:
        raise TypeError('At least two positional inputs are required.')

    def _outer_gen_2(x, y):
        x, y = np.asarray(x), np.asarray(y)
        x = x.reshape(x.shape + tuple(np.ones(y.ndim, dtype=int)))  # add singleton dimensions for broadcasting
        return x * y

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


def simplex_round(x):
    x = np.asarray(x)
    if x.min() < 0:
        raise ValueError("Input values must be non-negative.")
    elif x.sum() != 1:
        raise ValueError("Input values must sum to one.")

    out = np.zeros(x.size)
    up = 1
    for i, x_i in enumerate(x.flatten()):
        if x_i < up / 2:
            up -= x_i
        else:
            out[i] = 1
            break

    return out.reshape(x.shape)


#%% Plotting

def simplex_grid(n=1, shape=(1,), hull_mask=None):
    """
    Generate a uniform grid over a simplex.
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

    if n < sum(hull_mask.flatten()):
        raise ValueError("Input 'n' must meet or exceed the number of True values in 'hull_mask'.")


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

