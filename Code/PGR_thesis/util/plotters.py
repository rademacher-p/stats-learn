import numpy as np


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
