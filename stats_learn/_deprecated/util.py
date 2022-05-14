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


def vectorize_first_arg(func):
    @wraps(func)
    def func_wrap(*args, **kwargs):
        if isinstance(args[0], Iterable):
            return list(func(arg, *args[1:], **kwargs) for arg in args[0])
        else:
            return func(*args, **kwargs)

    return func_wrap


def empirical_pmf(d, supp, shape):
    """Generates the empirical PMF for a data set."""

    supp, supp_shape = check_data_shape(supp, shape)
    n_supp = math.prod(supp_shape)
    supp_flat = supp.reshape(n_supp, -1)

    if d.size == 0:
        return np.zeros(supp_shape)

    d, _set_shape = check_data_shape(d, shape)
    n = math.prod(_set_shape)
    d_flat = d.reshape(n, -1)

    dist = np.zeros(n_supp)
    for d_i in d_flat:
        eq_supp = np.all(d_i.flatten() == supp_flat, axis=-1)
        if eq_supp.sum() != 1:
            raise ValueError("Data must be in the support.")

        dist[eq_supp] += 1

    return dist.reshape(supp_shape) / n
