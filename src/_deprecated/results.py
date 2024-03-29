import math
from itertools import product
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from more_itertools import all_equal

from stats_learn.bayes import models as bayes_models
from stats_learn.util.base import check_data_shape

# TODO: LOTS of D.R.Y. fixes can be performed!!


def combined_compare(
    predictors,
    model,
    params=None,
    n_train=0,
    n_test=0,
    n_mc=1,
    x=None,
    stats=None,
    verbose=False,
):
    # uses Welford's online algorithm https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]
    n_train = np.sort(n_train)

    if stats is None:
        stats = set()
    else:
        stats = set(stats)
    if "std" in stats:
        stats.add("cov")
    if "cov" in stats:
        stats.add("mean")

    do_stats = "mean" in stats
    do_loss = n_test > 0

    if do_stats:
        if x is None:
            # space_x = check_spaces_x(predictors)
            space_x = model.space["x"]
            x = space_x.x_plt

        shape, size, ndim = model.shape, model.size, model.ndim
        x, set_shape = check_data_shape(x, shape["x"])

    # Initialize arrays
    loss_full = []
    for params in params_full:
        params_shape = tuple(len(vals) for _, vals in params.items())
        # loss = np.empty((n_mc, len(n_train_delta), *params_shape))
        if do_loss:
            loss = np.zeros((len(n_train), *params_shape))
        else:
            loss = np.full((len(n_train), *params_shape), np.nan)
        loss_full.append(loss)

    _samp, dtype = [], []
    for stat in stats:
        if stat in {"mode", "median", "mean"}:
            stat_shape = set_shape + shape["y"]
        elif stat in {"std", "cov"}:
            stat_shape = set_shape + 2 * shape["y"]
        else:
            raise ValueError
        _samp.append(np.zeros(stat_shape))
        dtype.append((stat, np.float64, stat_shape))
        # TODO: dtype float? need model dtype attribute?!

    temp = np.array(tuple(_samp), dtype=dtype)
    y_stats_full = [
        np.tile(temp, reps=(len(n_train), *loss.shape[1:])) for loss in loss_full
    ]

    # Generate random data and make predictions
    def _update_stats(array, y, i):
        stats_ = array.dtype.names
        if "mean" in stats_:
            _mean_prev = array["mean"]
            array["mean"] += (y - _mean_prev) / (i + 1)
            if "cov" in stats_:
                _temp_1 = (y - _mean_prev).reshape(math.prod(set_shape), size["y"])
                _temp_2 = (y - array["mean"]).reshape(math.prod(set_shape), size["y"])
                _temp = np.array(
                    [np.tensordot(t1, t2, 0) for t1, t2 in zip(_temp_1, _temp_2)]
                )
                array["cov"] += _temp.reshape(set_shape + 2 * shape["y"])

    for i_mc in range(n_mc):
        if verbose:
            print(f"Stats/loss iteration: {i_mc + 1}/{n_mc}", end="\r")

        d = model.rvs(n_test + n_train[-1])
        d_test, d_train = d[:n_test], d[n_test:]

        for predictor, params, y_stats, loss in zip(
            predictors, params_full, y_stats_full, loss_full
        ):
            for i_n in range(len(n_train)):
                if i_n == 0 or not predictor.can_warm_start:
                    slice_ = slice(0, n_train[i_n])
                    warm_start = False  # resets learner for new iteration
                else:  # fit with incremental data partitions
                    slice_ = slice(n_train[i_n - 1], n_train[i_n])
                    warm_start = True

                predictor.fit(d_train[slice_], warm_start=warm_start)

                if len(params) == 0:
                    if do_stats:
                        y_mc = predictor.predict(x)
                        _update_stats(y_stats[i_n], y_mc, i_mc)

                    if do_loss:
                        loss[i_n] += predictor.evaluate(d_test)
                else:
                    for i_v, param_vals in enumerate(list(product(*params.values()))):
                        predictor.set_params(**dict(zip(params.keys(), param_vals)))

                        idx = (i_n, *np.unravel_index(i_v, loss.shape[1:]))

                        if do_stats:
                            y_mc = predictor.predict(x)
                            _update_stats(y_stats[idx], y_mc, i_mc)

                        if do_loss:
                            loss[idx] += predictor.evaluate(d_test)

    if "cov" in stats:
        for y_stats in y_stats_full:
            y_stats["cov"] /= n_mc
    if "std" in stats:
        for y_stats in y_stats_full:
            y_stats["std"] = np.sqrt(y_stats["cov"])

    # loss_full = [loss.mean() for loss in loss_full]
    loss_full = [loss / n_mc for loss in loss_full]

    # Print results as Markdown table
    if verbose:
        _print_risk(predictors, params_full, n_train, loss_full, file=None)

    return y_stats_full, loss_full


def plot_fit_compare(predictors, d, params=None, ax=None):
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if ax is None:
        # use first predictors space by default
        ax = predictors[0].space["x"].make_axes()

    ax.scatter(d["x"], d["y"], c="k", marker=".", label=None)

    for predictor, params in zip(predictors, params_full):
        predictor.fit(d)
        if len(params) == 0:
            predictor.plot_predict(ax=ax, label=predictor.name)
        elif len(params) == 1:
            param_name, param_vals = list(params.items())[0]
            labels = [
                f"{predictor.name}, {predictor.tex_params(param_name, val)}"
                for val in param_vals
            ]
            for param_val, label in zip(param_vals, labels):
                predictor.set_params(**{param_name: param_val})
                predictor.plot_predict(ax=ax, label=label)
        else:
            raise NotImplementedError(
                "Only up to one varying parameter currently supported."
            )

        # predictor.plot_predict(ax=ax, label=predictor.name)

    if len(predictors) > 1:
        ax.legend()
    else:
        ax.set(title=predictors[0].name)


def predict_stats_compare(
    predictors,
    model,
    params=None,
    n_train=0,
    n_mc=1,
    x=None,
    stats=("mode",),
    verbose=False,
):
    y_stats_full, __ = combined_compare(
        predictors,
        model,
        params,
        n_train,
        n_test=0,
        n_mc=n_mc,
        x=x,
        stats=stats,
        verbose=verbose,
    )

    return y_stats_full

    # # uses Welford's online algorithm https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    #
    # if x is None:
    #     # space_x = check_spaces_x(predictors)
    #     space_x = model.space['x']
    #     x = space_x.x_plt
    #
    # if params is None:
    #     params_full = [{} for _ in predictors]
    # else:
    #     params_full = [item if item is not None else {} for item in params]
    #
    # if isinstance(n_train, (Integral, np.integer)):
    #     n_train = [n_train]
    # n_train = np.sort(n_train)
    #
    # shape, size, ndim = model.shape, model.size, model.ndim
    # x, set_shape = check_data_shape(x, shape['x'])
    #
    # # Initialize arrays
    # params_shape_full = []
    # for params in params_full:
    #     params_shape = tuple(len(vals) for _, vals in params.items())
    #     params_shape_full.append(params_shape)
    #
    # stats = set(stats)
    # if 'std' in stats:
    #     stats.add('cov')
    # if 'cov' in stats:
    #     stats.add('mean')
    #
    # _samp, dtype = [], []
    # for stat in stats:
    #     if stat in {'mode', 'median', 'mean'}:
    #         stat_shape = set_shape + shape['y']
    #     elif stat in {'std', 'cov'}:
    #         stat_shape = set_shape + 2 * shape['y']
    #     else:
    #         raise ValueError
    #     _samp.append(np.zeros(stat_shape))
    #     dtype.append((stat, np.float64, stat_shape))  # TODO: dtype float? need model dtype attribute?!
    #
    # y_stats_full = [np.tile(np.array(tuple(_samp), dtype=dtype), reps=(len(n_train), *param_shape))
    #                 for param_shape in params_shape_full]
    #
    # def _update_stats(array, y, i):
    #     if 'mean' in stats:
    #         _mean_prev = array['mean']
    #         array['mean'] += (y - _mean_prev) / (i + 1)
    #         if 'cov' in stats:
    #             _temp_1 = (y - _mean_prev).reshape(math.prod(set_shape), size['y'])
    #             _temp_2 = (y - array['mean']).reshape(math.prod(set_shape), size['y'])
    #             _temp = np.array([np.tensordot(t1, t2, 0) for t1, t2 in zip(_temp_1, _temp_2)])
    #             array['cov'] += _temp.reshape(set_shape + 2 * shape['y'])
    #
    # # Generate random data and make predictions
    # for i_mc in range(n_mc):
    #     if verbose:
    #         print(f"Stats iteration: {i_mc + 1}/{n_mc}", end='\r')
    #
    #     d = model.rvs(n_train[-1])
    #     for predictor, params, params_shape, y_stats in zip(predictors, params_full, params_shape_full, y_stats_full):
    #         for i_n in range(len(n_train)):
    #             if i_n == 0 or not predictor.can_warm_start:
    #                 slice_ = slice(0, n_train[i_n])
    #                 warm_start = False  # resets learner for new iteration
    #             else:  # fit with incremental data partitions
    #                 slice_ = slice(n_train[i_n-1], n_train[i_n])
    #                 warm_start = True
    #
    #             predictor.fit(d[slice_], warm_start=warm_start)
    #
    #             if len(params) == 0:
    #                 y_mc = predictor.predict(x)
    #                 _update_stats(y_stats[i_n], y_mc, i_mc)
    #             else:
    #                 for i_v, param_vals in enumerate(list(product(*params.values()))):
    #                     predictor.set_params(**dict(zip(params.keys(), param_vals)))
    #                     y_mc = predictor.predict(x)
    #
    #                     idx = (i_n, *np.unravel_index(i_v, params_shape))
    #                     _update_stats(y_stats[idx], y_mc, i_mc)
    #
    # if 'cov' in stats:
    #     for y_stats in y_stats_full:
    #         y_stats['cov'] /= n_mc
    # if 'std' in stats:
    #     for y_stats in y_stats_full:
    #         y_stats['std'] = np.sqrt(y_stats['cov'])
    #
    # return y_stats_full


def _plot_stats(
    y_stats_full, space_x, predictors, params_full, n_train=0, x=None, ax=None
):
    if x is None:
        x = space_x.x_plt

    if ax is None:
        ax = space_x.make_axes()

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    names = y_stats_full[0].dtype.names
    if "mean" not in names:
        raise ValueError("Need mean in stats")
    do_std = "std" in names

    out = []
    if len(predictors) == 1:
        predictor, params, y_stats = predictors[0], params_full[0], y_stats_full[0]
        title = str(predictor.name)
        if len(params) == 0:
            if len(n_train) == 1:
                labels = [None]
                title += f", $N = {n_train[0]}$"
            else:
                labels = [f"$N = {n}$" for n in n_train]
        elif len(params) == 1:
            param_name, param_vals = list(params.items())[0]
            if len(n_train) == 1:
                y_stats = y_stats.squeeze(axis=0)
                title += f", $N = {n_train[0]}$"
                if len(param_vals) == 1:
                    labels = [None]
                    # title += f", ${param_name} = {param_vals[0]}$"
                    title += f", {predictor.tex_params(param_name, param_vals[0])}"
                else:
                    # labels = [f"${param_name} = {val}$" for val in param_vals]
                    labels = [
                        f"{predictor.tex_params(param_name, val)}" for val in param_vals
                    ]
            elif len(param_vals) == 1:
                y_stats = y_stats.squeeze(axis=1)
                labels = [f"$N = {n}$" for n in n_train]
                # title += f", ${param_name} = {param_vals[0]}$"
                title += f", {predictor.tex_params(param_name, param_vals[0])}"
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(
                "Only up to one varying parameter currently supported."
            )

        for y_stat, label in zip(y_stats, labels):
            y_mean = y_stat["mean"]
            y_std = y_stat["std"] if do_std else None
            plt_data = space_x.plot_xy(x, y_mean, y_std, ax=ax, label=label)
            out.append(plt_data)

        if labels != [None]:
            ax.legend()

    else:
        if len(n_train) == 1:
            # TODO: enumerates and kwargs for errorbar predict. Remove??
            # lens = [1 if len(p) == 0 else len(list(p.values())[0]) for p in params_full]
            # n_lines = sum(lens)

            title = f"$N = {n_train[0]}$"
            for predictor, params, y_stats in zip(
                predictors, params_full, y_stats_full
            ):
                if len(params) == 0:
                    labels = [predictor.name]
                elif len(params) == 1:
                    y_stats = y_stats.squeeze(0)
                    param_name, param_vals = list(params.items())[0]
                    # labels = [f"{predictor.name}, ${param_name} = {val}$" for val in param_vals]
                    labels = [
                        f"{predictor.name}, {predictor.tex_params(param_name, val)}"
                        for val in param_vals
                    ]
                else:
                    raise ValueError

                # for i_v, (y_stat, label) in enumerate(zip(y_stats, labels)):
                #     xy_kwargs = {}
                #     if isinstance(space_x, spaces.Discrete):
                #         xy_kwargs['errorevery'] = (sum(lens[:i_p]) + i_v, n_lines)
                #
                #     y_mean = y_stat['mean']
                #     y_std = y_stat['std'] if do_std else None
                #     plt_data = space_x.plot_xy(x, y_mean, y_std, ax=ax, label=label, **xy_kwargs)
                #     out.append(plt_data)

                for y_stat, label in zip(y_stats, labels):
                    y_mean = y_stat["mean"]
                    y_std = y_stat["std"] if do_std else None
                    plt_data = space_x.plot_xy(x, y_mean, y_std, ax=ax, label=label)
                    out.append(plt_data)
        else:
            raise ValueError(
                "Plotting not supported for multiple predictors and multiple values of n_train."
            )

        ax.legend()

    ax.set_title(title)

    return out


def plot_predict_stats_compare(
    predictors,
    model,
    params=None,
    n_train=0,
    n_mc=1,
    x=None,
    do_std=False,
    verbose=False,
    ax=None,
):
    # space_x = check_spaces_x(predictors)
    space_x = model.space["x"]
    if x is None:
        x = space_x.x_plt

    stats = ("mean", "std") if do_std else ("mean",)  # TODO: generalize for mode, etc.
    y_stats_full = predict_stats_compare(
        predictors, model, params, n_train, n_mc, x, stats, verbose
    )

    # if ax is None:
    #     ax = space_x.make_axes()

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    # if isinstance(n_train, (Integral, np.integer)):
    #     n_train = [n_train]

    return _plot_stats(y_stats_full, space_x, predictors, params_full, n_train, x, ax)

    # out = []
    # if len(predictors) == 1:
    #     predictor, params, y_stats = predictors[0], params_full[0], y_stats_full[0]
    #     title = str(predictor.name)
    #     if len(params) == 0:
    #         if len(n_train) == 1:
    #             labels = [None]
    #             title += f", $N = {n_train[0]}$"
    #         else:
    #             labels = [f"$N = {n}$" for n in n_train]
    #     elif len(params) == 1:
    #         param_name, param_vals = list(params.items())[0]
    #         if len(n_train) == 1:
    #             y_stats = y_stats.squeeze(axis=0)
    #             title += f", $N = {n_train[0]}$"
    #             if len(param_vals) == 1:
    #                 labels = [None]
    #                 # title += f", ${param_name} = {param_vals[0]}$"
    #                 title += f", {predictor.tex_params(param_name, param_vals[0])}"
    #             else:
    #                 # labels = [f"${param_name} = {val}$" for val in param_vals]
    #                 labels = [f"{predictor.tex_params(param_name, val)}" for val in param_vals]
    #         elif len(param_vals) == 1:
    #             y_stats = y_stats.squeeze(axis=1)
    #             labels = [f"$N = {n}$" for n in n_train]
    #             # title += f", ${param_name} = {param_vals[0]}$"
    #             title += f", {predictor.tex_params(param_name, param_vals[0])}"
    #         else:
    #             raise NotImplementedError
    #     else:
    #         raise NotImplementedError("Only up to one varying parameter currently supported.")
    #
    #     for y_stat, label in zip(y_stats, labels):
    #         y_mean = y_stat['mean']
    #         y_std = y_stat['std'] if do_std else None
    #         plt_data = space_x.plot_xy(x, y_mean, y_std, ax=ax, label=label)
    #         out.append(plt_data)
    #
    #     if labels != [None]:
    #         ax.legend()
    #
    # else:
    #     if len(n_train) == 1:
    #         # TODO: enumerates and kwargs for errorbar predict. Remove??
    #         # lens = [1 if len(p) == 0 else len(list(p.values())[0]) for p in params_full]
    #         # n_lines = sum(lens)
    #
    #         title = f'$N = {n_train[0]}$'
    #         for predictor, params, y_stats in zip(predictors, params_full, y_stats_full):
    #             if len(params) == 0:
    #                 labels = [predictor.name]
    #             elif len(params) == 1:
    #                 y_stats = y_stats.squeeze(0)
    #                 param_name, param_vals = list(params.items())[0]
    #                 # labels = [f"{predictor.name}, ${param_name} = {val}$" for val in param_vals]
    #                 labels = [f"{predictor.name}, {predictor.tex_params(param_name, val)}" for val in param_vals]
    #             else:
    #                 raise ValueError
    #
    #             # for i_v, (y_stat, label) in enumerate(zip(y_stats, labels)):
    #             #     xy_kwargs = {}
    #             #     if isinstance(space_x, spaces.Discrete):
    #             #         xy_kwargs['errorevery'] = (sum(lens[:i_p]) + i_v, n_lines)
    #             #
    #             #     y_mean = y_stat['mean']
    #             #     y_std = y_stat['std'] if do_std else None
    #             #     plt_data = space_x.plot_xy(x, y_mean, y_std, ax=ax, label=label, **xy_kwargs)
    #             #     out.append(plt_data)
    #
    #             for y_stat, label in zip(y_stats, labels):
    #                 y_mean = y_stat['mean']
    #                 y_std = y_stat['std'] if do_std else None
    #                 plt_data = space_x.plot_xy(x, y_mean, y_std, ax=ax, label=label)
    #                 out.append(plt_data)
    #     else:
    #         raise ValueError("Plotting not supported for multiple predictors and multiple values of n_train.")
    #
    #     ax.legend()
    #
    # ax.set_title(title)
    #
    # return out


def _print_risk(predictors, params, n_train, losses, file=None):
    title = ""
    index_n = pd.Index(n_train, name="N")
    if len(predictors) == 1:
        predictor, param, loss = predictors[0], params[0], losses[0]
        if len(param) == 0:
            df = pd.DataFrame(loss, index_n, columns=[predictor.name])
            # df = pd.Series(loss, index_n, name=predictor.name)
        elif len(param) == 1:
            param_name, param_vals = list(param.items())[0]
            index_param = param_vals
            title = f"{predictor.name}, varying {param_name}"
            # index_param = pd.Index(param_vals, name=param_name)
            # title = predictor.name
            df = pd.DataFrame(loss, index_n, columns=index_param)
        else:
            raise NotImplementedError(
                "Only up to one varying parameter currently supported."
            )
    else:
        data = []
        columns = []
        for predictor, param, loss in zip(predictors, params, losses):
            if len(param) == 0:
                data.append(loss[..., np.newaxis])
                columns.append(predictor.name)
            elif len(param) == 1:
                data.append(loss)
                param_name, param_vals = list(param.items())[0]
                columns.extend(
                    [
                        f"{predictor.name}, {predictor.tex_params(param_name, val)}"
                        for val in param_vals
                    ]
                )
            else:
                raise NotImplementedError(
                    "Only up to one varying parameter currently supported."
                )

        data = np.concatenate(data, axis=1)
        df = pd.DataFrame(data, index_n, columns)

    df = df.transpose()

    print(title)
    print(df.to_markdown(tablefmt="github", floatfmt=".3f"), file=file)


def risk_eval_sim_compare(
    predictors, model, params=None, n_train=0, n_test=1, n_mc=1, verbose=False
):
    __, loss_full = combined_compare(
        predictors, model, params, n_train, n_test, n_mc, verbose=verbose
    )

    return loss_full

    # if params is None:
    #     params_full = [{} for _ in predictors]
    # else:
    #     params_full = [item if item is not None else {} for item in params]
    #
    # if isinstance(n_train, (Integral, np.integer)):
    #     n_train = [n_train]
    # n_train = np.sort(n_train)
    #
    # loss_full = []
    # for params in params_full:
    #     params_shape = tuple(len(vals) for _, vals in params.items())
    #     # loss = np.empty((n_mc, len(n_train_delta), *params_shape))
    #     loss = np.zeros((len(n_train), *params_shape))
    #     loss_full.append(loss)
    #
    # for i_mc in range(n_mc):
    #     if verbose:
    #         print(f"Loss iteration: {i_mc + 1}/{n_mc}", end='\r')
    #
    #     d = model.rvs(n_test + n_train[-1])
    #     d_test, d_train = d[:n_test], d[n_test:]
    #
    #     for predictor, params, loss in zip(predictors, params_full, loss_full):
    #         for i_n in range(len(n_train)):
    #             if i_n == 0 or not predictor.can_warm_start:
    #                 slice_ = slice(0, n_train[i_n])
    #                 warm_start = False  # resets learner
    #             else:  # fit with incremental data partitions
    #                 slice_ = slice(n_train[i_n-1], n_train[i_n])
    #                 warm_start = True
    #
    #             predictor.fit(d_train[slice_], warm_start=warm_start)
    #
    #             if len(params) == 0:
    #                 loss[i_n] += predictor.evaluate(d_test)
    #             else:
    #                 for i_v, param_vals in enumerate(list(product(*params.values()))):
    #                     predictor.set_params(**dict(zip(params.keys(), param_vals)))
    #                     idx = (i_n, *np.unravel_index(i_v, loss.shape[1:]))
    #                     loss[idx] += predictor.evaluate(d_test)
    #
    # # loss_full = [loss.mean() for loss in loss_full]
    # loss_full = [loss / n_mc for loss in loss_full]
    #
    # # Print results as Markdown table
    # if verbose:
    #     _print_risk(predictors, params_full, n_train, loss_full, file=None)
    #
    # return loss_full


def risk_eval_comp_compare(
    predictors, model, params=None, n_train=0, n_test=1, verbose=False
):
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    loss_full = []
    for predictor, params in zip(predictors, params_full):
        if verbose:
            print(f"Predictor: {predictor.name}")

        if len(params) == 0:
            loss_full.append(predictor.evaluate_comp(model, n_train, n_test))
        else:
            params_shape = tuple(len(vals) for _, vals in params.items())
            loss = np.empty((len(n_train),) + params_shape)
            for i_v, param_vals in enumerate(list(product(*params.values()))):
                predictor.set_params(**dict(zip(params.keys(), param_vals)))

                idx_p = np.unravel_index(i_v, loss.shape[1:])
                idx = (np.arange(len(n_train)),) + tuple(
                    [k for _ in range(len(n_train))] for k in idx_p
                )
                loss[idx] = predictor.evaluate_comp(model, n_train, n_test)
                # loss[:, np.unravel_index(i_v, loss.shape[2:])] = predictor.evaluate_comp(model, n_train)

            loss_full.append(loss)

    # Print results as Markdown table
    if verbose:
        _print_risk(predictors, params_full, n_train, loss_full, file=None)

    return loss_full


def _plot_risk_eval_compare(
    losses, do_bayes, predictors, params=None, n_train=0, ax=None
):
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    if ax is None:
        _, ax = plt.subplots()
        if do_bayes:
            ylabel = r"$\mathcal{R}(f)$"
        else:
            ylabel = r"$\mathcal{R}_{\Theta}(f;\theta)$"
        ax.set(ylabel=ylabel)

    out = []
    if len(predictors) == 1:
        predictor, params, loss = predictors[0], params_full[0], losses[0]
        title = str(predictor.name)

        if len(params) == 0:
            loss = loss[np.newaxis]
            xlabel, x_plt = "$N$", n_train
            labels = [None]
        elif len(params) == 1:
            param_name, param_vals = list(params.items())[0]
            if len(n_train) < len(param_vals):
                xlabel, x_plt = predictor.tex_params(param_name), param_vals
                if len(n_train) == 1:
                    title += f", $N = {n_train[0]}$"
                    labels = [None]
                else:
                    labels = [f"$N = {n}$" for n in n_train]
            else:
                loss = np.transpose(loss)
                xlabel, x_plt = "$N$", n_train
                if len(param_vals) == 1:
                    # title += f", {param_name} = {param_vals[0]}"
                    title += f", {predictor.tex_params(param_name, param_vals[0])}"
                    labels = [None]
                else:
                    # labels = [f"{param_name} = {val}" for val in param_vals]
                    labels = [
                        f"{predictor.tex_params(param_name, val)}" for val in param_vals
                    ]
        else:
            raise NotImplementedError(
                "Only up to one varying parameter currently supported."
            )

        for loss_plt, label in zip(loss, labels):
            plt_data = ax.plot(x_plt, loss_plt, label=label)
            out.append(plt_data)

        if labels != [None]:
            ax.legend()
    else:
        if (
            all_equal(params.keys() for params in params_full)
            and len(n_train) == 1
            and len(params_full[0]) == 1
        ):
            # Plot versus parameter for multiple predictors of same type

            title = f"$N = {n_train[0]}$"
            xlabel = predictors[0].tex_params(list(params_full[0].keys())[0])

            for predictor, params, loss in zip(predictors, params_full, losses):
                x_plt = list(params.values())[0]

                loss_plt = loss[0]
                label = predictor.name

                plt_data = ax.plot(x_plt, loss_plt, label=label)

                out.append(plt_data)

                ax.legend()

        else:
            title = ""
            xlabel, x_plt = "$N$", n_train
            for predictor, params, loss in zip(predictors, params_full, losses):
                if len(params) == 0:
                    loss = loss[np.newaxis]
                    labels = [predictor.name]
                elif len(params) == 1:
                    loss = np.transpose(loss)
                    param_name, param_vals = list(params.items())[0]
                    # labels = [f"{predictor.name}, {param_name} = {val}" for val in param_vals]
                    labels = [
                        f"{predictor.name}, {predictor.tex_params(param_name, val)}"
                        for val in param_vals
                    ]
                else:
                    raise NotImplementedError(
                        "Only up to one varying parameter currently supported."
                    )

                for loss_plt, label in zip(loss, labels):
                    plt_data = ax.plot(x_plt, loss_plt, label=label)
                    out.append(plt_data)

                ax.legend()

    ax.set(xlabel=xlabel)
    ax.set_title(title)

    return out


def plot_risk_eval_sim_compare(
    predictors, model, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, ax=None
):
    losses = risk_eval_sim_compare(
        predictors, model, params, n_train, n_test, n_mc, verbose
    )
    do_bayes = isinstance(model, bayes_models.Base)
    return _plot_risk_eval_compare(losses, do_bayes, predictors, params, n_train, ax)


def plot_risk_eval_comp_compare(
    predictors, model, params=None, n_train=0, n_test=1, verbose=False, ax=None
):
    losses = risk_eval_comp_compare(predictors, model, params, n_train, n_test, verbose)
    do_bayes = isinstance(model, bayes_models.Base)
    return _plot_risk_eval_compare(losses, do_bayes, predictors, params, n_train, ax)


def plot_risk_disc(
    predictors, model, params=None, n_train=0, n_test=1, n_mc=500, verbose=True, ax=None
):
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if not all_equal(params_full):
        raise ValueError
    # TODO: check models for equality

    losses = risk_eval_sim_compare(
        predictors, model, params, n_train, n_test, n_mc, verbose
    )

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    if ax is None:
        _, ax = plt.subplots()
        if isinstance(model, bayes_models.Base):
            ylabel = r"$\mathcal{R}(f)$"
        else:
            ylabel = r"$\mathcal{R}_{\Theta}(f;\theta)$"
        ax.set(ylabel=ylabel)

    loss = np.stack(losses, axis=-1)
    params = params_full[0]

    x_plt = np.array(
        [len(pr.model.space["x"].values) for pr in predictors]
    )  # discretization set size
    # title = str(predictors[0].name)
    title = r"$\mathrm{Dir}$"

    out = []
    if len(params) == 0:
        if len(n_train) == 1:
            title += f", $N = {n_train[0]}$"
            labels = [None]
        else:
            labels = [f"$N = {n}$" for n in n_train]

    elif len(params) == 1:
        param_name, param_vals = list(params.items())[0]

        if len(n_train) > 1 and len(param_vals) == 1:
            loss = loss.squeeze(axis=1)
            title += f", {predictors[0].tex_params(param_name, param_vals[0])}"
            labels = [f"$N = {n}$" for n in n_train]
        elif len(n_train) == 1 and len(param_vals) > 1:
            loss = loss.squeeze(axis=0)
            title += f", $N = {n_train[0]}$"
            labels = [
                f"{predictors[0].tex_params(param_name, val)}" for val in param_vals
            ]
        else:
            raise ValueError

    else:
        raise ValueError

    for loss_plt, label in zip(loss, labels):
        plt_data = ax.plot(x_plt, loss_plt, label=label, marker=".")
        out.append(plt_data)

    if labels != [None]:
        ax.legend()

    ax.set(xlabel=r"$|\mathcal{T}|$")
    ax.set_title(title)

    return out


# %% Before warm-start fitting

# def predict_stats_compare(predictors, model, params=None, x=None, n_train=0, n_mc=1, stats=('mode',), verbose=False):
#
#     space_x = check_spaces_x(predictors)
#     if x is None:
#         x = space_x.x_plt
#
#     if params is None:
#         params_full = [{} for _ in predictors]
#     else:
#         params_full = [item if item is not None else {} for item in params]
#
#     if isinstance(n_train, (Integral, np.integer)):
#         n_train = [n_train]
#
#     shape, size, ndim = model.shape, model.size, model.ndim
#     x, set_shape = check_data_shape(x, shape['x'])
#     n_train_delta = np.diff(np.concatenate(([0], list(n_train))))
#
#     # Generate random data and make predictions
#     params_shape_full = []
#     y_full = []
#     for params in params_full:
#         params_shape = tuple(len(vals) for _, vals in params.items())
#         y = np.empty((n_mc, len(n_train_delta)) + params_shape + set_shape + shape['y'])
#         params_shape_full.append(params_shape)
#         y_full.append(y)
#
#     for i_mc in range(n_mc):
#         if verbose:
#             print(f"Stats iteration: {i_mc + 1}/{n_mc}")
#
#         d = model.rvs(n_train_delta.sum())
#         d_iter = np.split(d, np.cumsum(n_train_delta)[:-1])
#         for i_n, d in enumerate(d_iter):
#             warm_start = i_n > 0  # resets learner for new iteration
#             for predictor, params, params_shape, y in zip(predictors, params_full, params_shape_full, y_full):
#                 predictor.fit(d, warm_start=warm_start)
#                 if len(params) == 0:
#                     y[i_mc, i_n] = predictor.predict(x)
#                 else:
#                     for i_v, param_vals in enumerate(list(product(*params.values()))):
#                         predictor.set_params(**dict(zip(params.keys(), param_vals)))
#                         # params_shape = y.shape[2:-(len(set_shape) + ndim['y'])]
#                         y[i_mc, i_n][np.unravel_index(i_v, params_shape)] = predictor.predict(x)
#
#     # Generate statistics
#     _samp, dtype = [], []
#     for stat in stats:
#         if stat in {'mode', 'median', 'mean'}:
#             stat_shape = set_shape + shape['y']
#         elif stat in {'std', 'cov'}:
#             stat_shape = set_shape + 2 * shape['y']
#         else:
#             raise ValueError
#         _samp.append(np.empty(stat_shape))
#         dtype.append((stat, np.float64, stat_shape))  # TODO: dtype float? need model dtype attribute?!
#
#     y_stats_full = [np.tile(np.array(tuple(_samp), dtype=dtype), reps=(len(n_train_delta), *param_shape))
#                     for param_shape in params_shape_full]
#
#     for y, y_stats in zip(y_full, y_stats_full):
#         if 'mode' in stats:
#             y_stats['mode'] = np.mode(y, axis=0)
#
#         if 'median' in stats:
#             y_stats['median'] = np.median(y, axis=0)
#
#         if 'mean' in stats:
#             y_stats['mean'] = y.mean(axis=0)
#
#         if 'std' in stats:
#             if ndim['y'] == 0:
#                 y_stats['std'] = y.std(axis=0)
#             else:
#                 raise ValueError("Standard deviation is only supported for singular data shapes.")
#
#         if 'cov' in stats:
#             if size['y'] == 1:
#                 _temp = y.var(axis=0)
#             else:
#                 _temp = np.moveaxis(y.reshape((n_mc, math.prod(set_shape), size['y'])), 0, -1)
#                 _temp = np.array([np.cov(t) for t in _temp])
#
#             y_stats['cov'] = _temp.reshape(set_shape + 2 * shape['y'])
#
#     return y_stats_full


# %% Pre- warm-start functionality loops for predict and loss

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
