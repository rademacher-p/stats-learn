import numpy as np


# def diag_multi(x):
#     x = np.array(x)
#     i = np.unravel_index(range(x.size), x.shape)
#     out = np.zeros(2 * x.shape)
#     out[2 * i] = x.flatten()
#     return out

# def diag_multi(x):
#     x = np.array(x)
#     return np.diagflat(x).reshape(2 * x.shape)


# def outer_gen(*args):
#     args = tuple(map(np.asarray, args))
#     if len(args) == 1:
#         return args[0]
#
#     ix = np.ix_(*map(np.ravel, args))
#     _temp = functools.reduce(lambda x, y: x * y, ix)
#     return _temp.reshape(tuple(itertools.chain(*map(np.shape, args))))


# def outer_gen(*args):
#     n_args = len(args)
#     if n_args < 2:
#         return np.asarray(args[0])
#
#     def _outer_gen_2(x, y):
#         x, y = np.asarray(x), np.asarray(y)
#         x = x.reshape(x.shape + tuple(np.ones(y.ndim, dtype=int)))  # add singleton dimensions for broadcasting
#         return x * y
#
#     out = args[0]
#     for arg in args[1:]:
#         out = _outer_gen_2(out, arg)
#     return out


# def inverse(x):
#     x = np.array(x)
#     if x.shape == ():
#         return 1 / x
#     else:
#         return np.linalg.inv(x)
#
#
# def determinant(x):
#     x = np.array(x)
#     if x.shape == ():
#         return x
#     else:
#         return np.linalg.det(x)
#
#
# def inner_prod(x, y, w=None):       # TODO: decompose weight and do colored norm?
#     x, y = np.array(x), np.array(y)
#
#     # if x.shape == () and y.shape == ():
#     #     if w is None:
#     #         return x * y
#     #     else:
#     #         return x * np.array(w) * y
#     if x.ndim == 0 or y.ndim == 0:
#         if w is None:
#             return x * y
#         else:
#             w = np.array(w)
#             if w.ndim != 0:
#                 raise ValueError
#             else:
#                 return x * np.array(w) * y
#     else:
#         if x.shape[0] != y.shape[0]:
#             raise ValueError("Inputs must have the same leading shape value.")
#
#         if w is None:
#             return np.moveaxis(x, 0, -1) @ y
#         else:
#             w = np.array(w)
#             if w.shape == () and x.ndim == 1 and y.ndim == 1:
#                 return x[..., np.newaxis] * w * y
#             elif w.shape != 2 * x.shape[:1]:
#                 raise ValueError(f"Weighting matrix must have shape {2 * x.shape[0]}.")
#             else:
#                 return np.moveaxis(x, 0, -1) @ w @ y


def simplex_round(x):
    x = np.array(x)
    if np.min(x) < 0:
        raise ValueError("Input values must be non-negative.")
    elif not np.isclose(x.sum(), 1):
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


def prob_disc(shape):
    """Create (unnormalized) probability array for a discretization grid. Lower edge/corner probabilities."""

    p = np.ones(shape)
    idx = np.nonzero(p)
    n = np.zeros(p.size)
    for i, size in zip(idx, shape):
        n += np.all([i > 0, i < size-1], axis=0)
    p[idx] = 2**n
    return p


def main():
    print(prob_disc((4, 3)))
