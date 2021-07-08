import math
from itertools import product
from numbers import Integral

import numpy as np

from stats_learn.util.base import check_data_shape
from stats_learn.util.spaces import check_spaces_x


#%% Before warm-start fitting
def predict_stats_compare(predictors, model, params=None, x=None, n_train=0, n_mc=1, stats=('mode',), verbose=False):

    space_x = check_spaces_x(predictors)
    if x is None:
        x = space_x.x_plt

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    shape, size, ndim = model.shape, model.size, model.ndim
    x, set_shape = check_data_shape(x, shape['x'])
    n_train_delta = np.diff(np.concatenate(([0], list(n_train))))

    # Generate random data and make predictions
    params_shape_full = []
    y_full = []
    for params in params_full:
        params_shape = tuple(len(vals) for _, vals in params.items())
        y = np.empty((n_mc, len(n_train_delta)) + params_shape + set_shape + shape['y'])
        params_shape_full.append(params_shape)
        y_full.append(y)

    for i_mc in range(n_mc):
        if verbose:
            print(f"Stats iteration: {i_mc + 1}/{n_mc}")

        d = model.rvs(n_train_delta.sum())
        d_iter = np.split(d, np.cumsum(n_train_delta)[:-1])
        for i_n, d in enumerate(d_iter):
            warm_start = i_n > 0  # resets learner for new iteration
            for predictor, params, params_shape, y in zip(predictors, params_full, params_shape_full, y_full):
                predictor.fit(d, warm_start=warm_start)
                if len(params) == 0:
                    y[i_mc, i_n] = predictor.predict(x)
                else:
                    for i_v, param_vals in enumerate(list(product(*params.values()))):
                        predictor.set_params(**dict(zip(params.keys(), param_vals)))
                        # params_shape = y.shape[2:-(len(set_shape) + ndim['y'])]
                        y[i_mc, i_n][np.unravel_index(i_v, params_shape)] = predictor.predict(x)

    # Generate statistics
    _samp, dtype = [], []
    for stat in stats:
        if stat in {'mode', 'median', 'mean'}:
            stat_shape = set_shape + shape['y']
        elif stat in {'std', 'cov'}:
            stat_shape = set_shape + 2 * shape['y']
        else:
            raise ValueError
        _samp.append(np.empty(stat_shape))
        dtype.append((stat, np.float64, stat_shape))  # TODO: dtype float? need model dtype attribute?!

    y_stats_full = [np.tile(np.array(tuple(_samp), dtype=dtype), reps=(len(n_train_delta), *param_shape))
                    for param_shape in params_shape_full]

    for y, y_stats in zip(y_full, y_stats_full):
        if 'mode' in stats:
            y_stats['mode'] = np.mode(y, axis=0)

        if 'median' in stats:
            y_stats['median'] = np.median(y, axis=0)

        if 'mean' in stats:
            y_stats['mean'] = y.mean(axis=0)

        if 'std' in stats:
            if ndim['y'] == 0:
                y_stats['std'] = y.std(axis=0)
            else:
                raise ValueError("Standard deviation is only supported for singular data shapes.")

        if 'cov' in stats:
            if size['y'] == 1:
                _temp = y.var(axis=0)
            else:
                _temp = np.moveaxis(y.reshape((n_mc, math.prod(set_shape), size['y'])), 0, -1)
                _temp = np.array([np.cov(t) for t in _temp])

            y_stats['cov'] = _temp.reshape(set_shape + 2 * shape['y'])

    return y_stats_full


#%% Pre- warm-start functionality loops for predict and loss

# d = model.rvs(n_train[-1])
# d_iter = np.split(d, n_train[:-1])
# for i_n, d_n in enumerate(d_iter):
#     warm_start = i_n > 0  # resets learner for new iteration
#     for predictor, params, params_shape, y_stats in zip(predictors, params_full, params_shape_full,
#                                                         y_stats_full):
#
#         predictor.fit(d_n, warm_start=warm_start)
#
#         if len(params) == 0:
#             y_mc = predictor.predict(x)
#             _update_stats(y_stats[i_n], y_mc)
#         else:
#             for i_v, param_vals in enumerate(list(product(*params.values()))):
#                 predictor.set_params(**dict(zip(params.keys(), param_vals)))
#                 y_mc = predictor.predict(x)
#
#                 idx = (i_n, *np.unravel_index(i_v, params_shape))
#                 _update_stats(y_stats[idx], y_mc)

# d = model.rvs(n_test + n_train[-1])
# d_test, _d_train = d[:n_test], d[n_test:]
# d_train_iter = np.split(_d_train, n_train[:-1])
#
# for i_n, d_train in enumerate(d_train_iter):
#     warm_start = i_n > 0  # resets learner for new iteration
#     for predictor, params, loss in zip(predictors, params_full, loss_full):
#         predictor.fit(d_train, warm_start=warm_start)
#
#         if len(params) == 0:
#             # loss[i_mc, i_n] = predictor.evaluate(d_test)
#             loss[i_n] += predictor.evaluate(d_test)
#         else:
#             for i_v, param_vals in enumerate(list(product(*params.values()))):
#                 predictor.set_params(**dict(zip(params.keys(), param_vals)))
#                 # loss[i_mc, i_n][np.unravel_index(i_v, loss.shape[2:])] = predictor.evaluate(d_test)
#                 loss[i_n][np.unravel_index(i_v, loss.shape[1:])] += predictor.evaluate(d_test)
