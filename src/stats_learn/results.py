"""Assessment tools for learning and prediction performance."""

import inspect
import logging
import math
import pickle
import sys
from collections.abc import Collection
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from more_itertools import all_equal
from tqdm import trange

from stats_learn import bayes
from stats_learn.util import check_data_shape

if plt.rcParams["text.usetex"] and "upgreek" in plt.rcParams["text.latex.preamble"]:
    str_risk_bayes = r"$R_\uptheta(f)$"
else:
    str_risk_bayes = r"$R_\theta(f)$"


pickle_figs = False
log_fmt = "\n# %(asctime)s\n%(message)s\n"
date_fmt = "%Y-%m-%d %H:%M:%S"
file_log_mode = "a"

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
out_handler = logging.StreamHandler(stream=sys.stdout)
out_formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
out_handler.setFormatter(out_formatter)
logger.addHandler(out_handler)


@contextmanager
def _file_logger(file, file_format):
    """Add temporary FileHandler to logger."""
    if file is not None:
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file, mode=file_log_mode)
        file_formatter = logging.Formatter(file_format, datefmt=date_fmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        yield logger
        logger.removeHandler(file_handler)
    else:
        yield logger


def _log_and_fig(message, log_path, ax, img_path):
    """Save figure, add figure to message format and log."""
    file_format = log_fmt
    if img_path is not None:
        img_path = Path(img_path)
        img_path.parent.mkdir(parents=True, exist_ok=True)

        fig = ax.figure
        fig.savefig(img_path)
        if pickle_figs:
            mpl_file = img_path.parent / f"{img_path.stem}.mpl"
            with open(mpl_file, "wb") as f:
                pickle.dump(fig, f)

        if log_path is not None:
            try:
                img_path_rel = img_path.relative_to(Path(log_path).parent)
                img_path_png = img_path_rel.parent / f"{img_path_rel.stem}.png"
                file_format += f"\n![]({img_path_png.as_posix()})\n"
                fig.savefig(img_path.parent / f"{img_path.stem}.png")
            except ValueError:
                pass

    with _file_logger(log_path, file_format) as logger_:
        logger_.info(message)


# Printing/plotting helper functions
def _print_risk(predictors, params, n_train, losses):
    """Create Markdown format table of empirical risk for various predictors."""
    title = ""
    index_n = pd.Index(n_train, name="N")
    if len(predictors) == 1:
        predictor, param, loss = predictors[0], params[0], losses[0]
        if len(param) == 0:
            df = pd.DataFrame(loss, index_n, columns=[predictor.name])
        elif len(param) == 1:
            param_name, param_vals = list(param.items())[0]
            index_param = param_vals
            title += f"{predictor.name}, varying {param_name}\n\n"
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
                        f"{predictor.name}, {predictor.make_label(param_name, value)}"
                        for value in param_vals
                    ]
                )
            else:
                raise NotImplementedError(
                    "Only up to one varying parameter currently supported."
                )

        data = np.concatenate(data, axis=1)
        df = pd.DataFrame(data, index_n, columns)

    df = df.transpose()
    str_table = df.to_markdown(tablefmt="github", floatfmt=".3f")

    return title + str_table


def _plot_predict_stats(
    y_stats_full,
    space_x,
    predictors,
    params,
    n_train: int | Collection = 0,
    x=None,
    ax=None,
    ax_kwargs=None,
):
    """Plot prediction statistics for various predictors and parameterizations."""
    if x is None:
        x = space_x.x_plt

    if ax is None:
        if ax_kwargs is None:
            ax_kwargs = {}
        ax = space_x.make_axes(**ax_kwargs)

    params_full: list[dict]
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    n_train = np.array(n_train).reshape(-1)

    names = y_stats_full[0].dtype.names
    if "mean" not in names:
        raise ValueError("Need mean in stats")
    do_std = "std" in names

    labels: list[str | None]
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
                    title += f", {predictor.make_label(param_name, param_vals[0])}"
                else:
                    labels = [
                        f"{predictor.make_label(param_name, value)}"
                        for value in param_vals
                    ]
            elif len(param_vals) == 1:
                y_stats = y_stats.squeeze(axis=1)
                labels = [f"$N = {n}$" for n in n_train]
                title += f", {predictor.make_label(param_name, param_vals[0])}"
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
            # lens = [1 if len(p) == 0 else len(list(p.values())[0])
            #         for p in params_full]
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
                    labels = [
                        f"{predictor.name}, {predictor.make_label(param_name, value)}"
                        for value in param_vals
                    ]
                else:
                    raise ValueError

                # for i_v, (y_stat, label) in enumerate(zip(y_stats, labels)):
                #     xy_kwargs = {}
                #     if isinstance(space_x, spaces.Discrete):
                #         xy_kwargs["errorevery"] = (sum(lens[:i_p]) + i_v, n_lines)

                #     y_mean = y_stat["mean"]
                #     y_std = y_stat["std"] if do_std else None
                #     plt_data = space_x.plot_xy(
                #         x, y_mean, y_std, ax=ax, label=label, **xy_kwargs
                #     )
                #     out.append(plt_data)

                for y_stat, label in zip(y_stats, labels):
                    y_mean = y_stat["mean"]
                    y_std = y_stat["std"] if do_std else None
                    plt_data = space_x.plot_xy(x, y_mean, y_std, ax=ax, label=label)
                    out.append(plt_data)
        else:
            raise ValueError(
                "Plotting not supported for >1 predictors and >1 values of n_train."
            )

        ax.legend()

    ax.set_title(title)

    return out


def _plot_risk_eval_compare(
    losses,
    predictors,
    params=None,
    n_train: int | Collection = 0,
    ax=None,
    ax_kwargs=None,
):
    """Plot empirical risk for various predictors and parameterizations."""
    params_full: list[dict]
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    n_train = np.array(n_train).reshape(-1)

    if ax is None:
        if ax_kwargs is None:
            ax_kwargs = {}
        with plt.rc_context({"axes.xmargin": 0}):
            _, ax = plt.subplots(subplot_kw=ax_kwargs)
        ax.set(ylabel=r"$R(f; \rho)$")

    labels: list[str | None]
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
                xlabel, x_plt = predictor.make_label(param_name), param_vals
                if len(n_train) == 1:
                    title += f", $N = {n_train[0]}$"
                    labels = [None]
                else:
                    labels = [f"$N = {n}$" for n in n_train]
            else:
                loss = np.transpose(loss)
                xlabel, x_plt = "$N$", n_train
                if len(param_vals) == 1:
                    title += f", {predictor.make_label(param_name, param_vals[0])}"
                    labels = [None]
                else:
                    labels = [
                        f"{predictor.make_label(param_name, value)}"
                        for value in param_vals
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
            xlabel = predictors[0].make_label(list(params_full[0].keys())[0])

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
                    labels = [
                        f"{predictor.name}, {predictor.make_label(param_name, value)}"
                        for value in param_vals
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


# Assessment tools
def evaluate(predictor, loss_func, d):
    """
    Evaluate predictor using test data.

    Parameters
    ----------
    predictor : stats_learn.predictors.Base
    loss_func : callable
    d : np.ndarray
        The test data.

    Returns
    -------
    float
        Empirical risk (i.e. average test loss).

    """
    h = predictor.predict(d["x"])
    losses = loss_func(h, d["y"])
    return losses.mean()


def evaluate_from_model(predictor, loss_func, model, n_test=1, n_mc=1, rng=None):
    """
    Evaluate predictor using test data randomly drawn from a given data model.

    Parameters
    ----------
    predictor : stats_learn.predictors.Base
    loss_func : callable
    model : stats_learn.random.models.Base
        Model for training data generation.
    n_test : int, optional
        Number of test samples.
    n_mc : int, optional
        Number of Monte Carlo simulation iterations.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    Returns
    -------
    float
        Empirical risk (i.e. average test loss).

    """
    model.rng = rng
    losses = np.empty(n_mc)
    for i_mc in range(n_mc):
        d = model.sample(n_test)
        losses[i_mc] = evaluate(predictor, loss_func, d)

    return losses.mean()


def data_assess(
    predictors,
    loss_func,
    d_train=None,
    d_test=None,
    params=None,
    x=None,
    verbose=False,
    plot_fit=False,
    log_path=None,
    img_path=None,
    ax=None,
    ax_kwargs=None,
):
    """
    Assess and compare various predictors using a single dataset.

    Parameters
    ----------
    predictors : Collection of stats_learn.predictors.Base
        Predictors to assess.
    loss_func : callable
    d_train : array_like, optional
        Training data.
    d_test : array_like, optional
        Testing data.
    params : Collection of dict, optional
        Predictor parameters to evaluate. Each element corresponds to an element of
        `predictors` and contains an optional dictionary mapping parameter names to
        various values. Outer product of each parameter array is assessed.
    x : array_like, optional
        Values of observed element to use for assessment of prediction statistics.
    verbose : bool, optional
        Enables iteration print-out.
    plot_fit : bool, optional
        Enables plotting of fit predictors.
    log_path : os.PathLike or str, optional
        File for saving printed loss table and image path in Markdown format.
    img_path : os.PathLike or str, optional
        Directory for saving generated images.
    ax : matplotlib.axes.Axes, optional
        Axes onto which stats/losses are plotted.
    ax_kwargs : dict, optional
        Keyworld arguments for Axes construction.

    Returns
    -------
    list of ndarray
        Empirical risk values for each predictor and parameterization.

    """
    # TODO: Make similar to `model_assess` signature? Params first?

    # if not all_equal(p.space for p in predictors):
    #     raise ValueError("All predictor spaces must be equivalent.")

    space = predictors[0].space  # use first predictors space by default

    if d_train is None:
        _dtype = [(c, space[c].dtype, space[c].shape) for c in "xy"]
        d_train = np.array([], dtype=_dtype)
    if d_test is None:
        d_test = np.array([], dtype=[(c, space[c].dtype, space[c].shape) for c in "xy"])

    n_train, n_test = map(len, (d_train, d_test))
    do_loss = n_test > 0

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    loss_full = []
    for params in params_full:
        params_shape = tuple(len(vals) for _, vals in params.items())
        if do_loss:
            loss = np.zeros((1, *params_shape))
        else:
            loss = np.full((1, *params_shape), np.nan)
        loss_full.append(loss)

    if x is None:
        x = space["x"].x_plt

    handles = []
    if plot_fit:
        if ax is None:
            if ax_kwargs is None:
                ax_kwargs = {}
            ax = space["x"].make_axes(**ax_kwargs)
        h_data = space["x"].plot_xy(
            d_train["x"],
            d_train["y"],
            ax=ax,
            marker="o",
            c="k",
            linestyle="",
            label="$D$",
        )
        handles.extend(h_data)

    for predictor, params, loss in zip(predictors, params_full, loss_full):
        if verbose:
            # print(f"  Predictor: {predictor.name}", end='\r')
            print(f"  Predictor: {predictor.name}")
            # TODO: make `verbose` int, add levels of control?

        predictor.fit(d_train)

        if len(params) == 0:
            if plot_fit:
                h = predictor.plot_predict(x, ax=ax, label=predictor.name)
                handles.extend(h)

            if do_loss:
                loss[0] += evaluate(predictor, loss_func, d_test)

        elif len(params) == 1:
            param_name, param_vals = list(params.items())[0]
            labels = [
                f"{predictor.name}, {predictor.make_label(param_name, value)}"
                for value in param_vals
            ]
            for i_v, (param_val, label) in enumerate(zip(param_vals, labels)):
                predictor.set_params(**{param_name: param_val})

                if plot_fit:
                    h = predictor.plot_predict(x, ax=ax, label=label)
                    handles.extend(h)

                if do_loss:
                    idx = (0, *np.unravel_index(i_v, loss.shape[1:]))
                    loss[idx] += evaluate(predictor, loss_func, d_test)
        else:
            raise NotImplementedError(
                "Only up to one varying parameter currently supported."
            )

    if plot_fit:
        title = f"$N = {n_train}$"
        if len(predictors) == 1 and len(params_full[0]) == 0:
            title = f"{predictors[0].name}, " + title
        else:
            ax.legend(handles=handles)
        ax.set_title(title)

    # Logging
    message = f"- Test samples: {n_test}"
    if do_loss:
        message += f"\n\n{_print_risk(predictors, params_full, [n_train], loss_full)}"

    _log_and_fig(message, log_path, ax, img_path)

    return loss_full


def model_assess(
    predictors,
    loss_func,
    model,
    params=None,
    n_train=0,
    n_test=0,
    n_mc=1,
    x=None,
    stats=None,
    verbose=False,
    plot_stats=False,
    plot_loss=False,
    print_loss=False,
    log_path=None,
    img_path=None,
    ax=None,
    ax_kwargs=None,
    rng=None,
):
    """
    Assess and compare various predictors via prediction statistics and empirical risk.

    Uses Monte Carlo simulation.

    Parameters
    ----------
    predictors : Collection of stats_learn.predictors.Base
        Predictors to assess.
    loss_func : callable
    model : stats_learn.random.models.Base or stats_learn.bayes.models.Base
        Data-generating model.
    params : Collection of dict, optional
        Predictor parameters to evaluate. Each element corresponds to an element of `
        predictors` and contains an optional dictionary mapping parameter names to
        various values. Outer product of each parameter array is assessed.
    n_train : int or Collection of int, optional
        Training data volume.
    n_test : int, optional
        Test data volume.
    n_mc : int, optional
        Number of Monte Carlo simulation iterations.
    x : array_like, optional
        Values of observed element to use for assessment of prediction statistics.
    stats : Collection of str, optional
        Names of the statistics to generate, e.g. 'mean', 'std', 'cov', 'mode', etc.
    verbose : bool, optional
        Enables iteration print-out.
    plot_stats : bool, optional
        Enables plotting of prediction statistics.
    plot_loss : bool, optional
        Enables plotting of average loss.
    print_loss : bool, optional
        Enables print-out of average loss table.
    log_path : os.PathLike or str, optional
        File for saving printed loss table and image path in Markdown format.
    img_path : os.PathLike or str, optional
        Directory for saving generated images.
    ax : matplotlib.axes.Axes, optional
        Axes onto which stats/losses are plotted.
    ax_kwargs : dict, optional
        Keyworld arguments for Axes construction.
    rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

    Returns
    -------
    list of ndarray
        One item per predictor with prediction statistics for each parameterization.
    list of ndarray
        One item per predictor with empirical risk values for each parameterization.

    Notes
    -----
    Uses Welford's online algorithm https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    """
    # TODO: add mode!!!
    # TODO: train/test loss results?

    # if not all_equal(p.space for p in predictors):
    #     raise ValueError("All predictor spaces must be equivalent.")

    if plot_stats and plot_loss:
        raise NotImplementedError(
            "Cannot plot prediction statistics and losses at once."
        )

    model.rng = rng
    # model.rng = RNGMix.make_rng(rng)

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    n_train = np.array(n_train).reshape(-1)
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
        space_x = model.space["x"]
        if x is None:
            x = space_x.x_plt

        shape, size, _ = model.shape, model.size, model.ndim
        x, set_shape = check_data_shape(x, shape["x"])

    # Initialize arrays
    loss_full = []
    for params in params_full:
        params_shape = tuple(len(vals) for _, vals in params.items())
        if do_loss:
            loss = np.zeros((len(n_train), *params_shape))
        else:
            loss = np.full((len(n_train), *params_shape), np.nan)
        loss_full.append(loss)

    samp, dtype = [], []
    for stat in stats:
        if stat in {"mode", "median", "mean"}:
            stat_shape = set_shape + shape["y"]
        elif stat in {"std", "cov"}:
            stat_shape = set_shape + 2 * shape["y"]
        else:
            raise ValueError
        samp.append(np.zeros(stat_shape))
        dtype.append((stat, np.float64, stat_shape))
        # TODO: dtype float? need model dtype attribute?!

    _temp = np.array(tuple(samp), dtype=dtype)
    y_stats_full = [
        np.tile(_temp, reps=(len(n_train), *loss.shape[1:])) for loss in loss_full
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

    for i_mc in trange(n_mc, desc="MC iteration", disable=not verbose):
        # if verbose:
        #     print(f"MC iteration: {i_mc + 1}/{n_mc}")

        d = model.sample(n_train[-1] + n_test)
        d_train, d_test = d[: n_train[-1]], d[n_train[-1] :]

        for predictor, params, y_stats, loss in zip(
            predictors, params_full, y_stats_full, loss_full
        ):
            # if verbose:
            #     # print(f"  Predictor: {predictor.name}", end='\r')
            #     print(f"  Predictor: {predictor.name}")
            #     # TODO: make `verbose` int, add levels of control?

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
                        loss[i_n] += evaluate(predictor, loss_func, d_test)
                else:
                    for i_v, param_vals in enumerate(product(*params.values())):
                        predictor.set_params(**dict(zip(params.keys(), param_vals)))

                        idx = (i_n, *np.unravel_index(i_v, loss.shape[1:]))

                        if do_stats:
                            y_mc = predictor.predict(x)
                            _update_stats(y_stats[idx], y_mc, i_mc)

                        if do_loss:
                            loss[idx] += evaluate(predictor, loss_func, d_test)

    if "cov" in stats:
        for y_stats in y_stats_full:
            y_stats["cov"] /= n_mc
    if "std" in stats:
        for y_stats in y_stats_full:
            y_stats["std"] = np.sqrt(y_stats["cov"])

    loss_full = [loss / n_mc for loss in loss_full]

    # Plot
    if do_stats and plot_stats:
        _plot_predict_stats(
            y_stats_full, space_x, predictors, params_full, n_train, x, ax, ax_kwargs
        )
        ax = plt.gca()
    elif do_loss and plot_loss:
        _plot_risk_eval_compare(
            loss_full, predictors, params_full, n_train, ax, ax_kwargs
        )
        ax = plt.gca()
        if isinstance(model, bayes.models.Base):
            ax.set(ylabel=str_risk_bayes)  # different notation for bayes risk
    else:
        img_path = None

    # Logging
    message = f"- Seed = {rng}\n- Test samples: {n_test}\n- MC iterations: {n_mc}"
    if do_loss and print_loss:
        message += f"\n\n{_print_risk(predictors, params_full, n_train, loss_full)}"

    _log_and_fig(message, log_path, ax, img_path)

    return y_stats_full, loss_full


# Additional utilities
def risk_eval_analytic(
    predictors, model, params=None, n_train=0, n_test=1, verbose=False
):
    """Assess various predictors using analytical risk calculation."""
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    n_train = np.array(n_train).reshape(-1)

    loss_full = []
    for predictor, params in zip(predictors, params_full):
        if verbose:
            print(f"Predictor: {predictor.name}")

        if len(params) == 0:
            loss_full.append(predictor.evaluate_analytic(model, n_train, n_test))
        else:
            params_shape = tuple(len(vals) for _, vals in params.items())
            loss = np.empty((len(n_train),) + params_shape)
            for i_v, param_vals in enumerate(list(product(*params.values()))):
                predictor.set_params(**dict(zip(params.keys(), param_vals)))

                idx_p = np.unravel_index(i_v, loss.shape[1:])
                idx = (np.arange(len(n_train)),) + tuple(
                    [k for _ in range(len(n_train))] for k in idx_p
                )
                loss[idx] = predictor.evaluate_analytic(model, n_train, n_test)

            loss_full.append(loss)

    # Print results as Markdown table
    if verbose:
        _print_risk(predictors, params_full, n_train, loss_full)

    return loss_full


def plot_risk_disc(
    predictors,
    loss_func,
    model,
    params=None,
    n_train=0,
    n_test=1,
    n_mc=1,
    verbose=True,
    ax=None,
):
    """Plot risk against discretization set cardinality."""
    # TODO: integrate logging code? Cleanup!

    n_t_iter = np.array(
        [
            inspect.getclosurevars(pr.proc_funcs["pre"][0]).nonlocals["vals"].size
            for pr in predictors
        ]
    )

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if all(len(params) == 0 for params in params_full):
        params = {}
    elif all(list(params.keys()) == ["alpha_0"] for params in params_full):
        a0_full = np.array([params["alpha_0"] for params in params_full])
        if all(all_equal(arr) for arr in a0_full.transpose()):
            params = {"alpha_0": a0_full[0]}
            # tex_map = lambda x: r"$\alpha_0 = " + f"{x}$"

            def tex_map(x):
                return r"$\alpha_0 = " + f"{x}$"

        else:
            a0_full_norm = a0_full / n_t_iter[..., np.newaxis]
            if all(all_equal(arr) for arr in a0_full_norm.transpose()):
                params = {"alpha_0": a0_full_norm[0]}
                # tex_map = lambda x: r"$\alpha_0 / |\mathcal{T}| = " + f"{x}$"

                def tex_map(x):
                    return r"$\alpha_0 / |\mathcal{T}| = " + f"{x}$"

            else:
                raise ValueError
    else:
        raise ValueError

    __, losses = model_assess(
        predictors, loss_func, model, params_full, n_train, n_test, n_mc, verbose
    )

    n_train = np.array(n_train).reshape(-1)

    if ax is None:
        _, ax = plt.subplots()
        if isinstance(model, bayes.models.Base):
            ylabel = str_risk_bayes
        else:
            ylabel = r"$R(f; \rho)$"
        ax.set(ylabel=ylabel)

    loss = np.stack(losses, axis=-1)

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
            # title += f", {predictors[0].make_label(param_name, param_vals[0])}"
            title += f", {tex_map(param_vals[0])}"
            labels = [f"$N = {n}$" for n in n_train]
        elif len(n_train) == 1 and len(param_vals) > 1:
            loss = loss.squeeze(axis=0)
            title += f", $N = {n_train[0]}$"
            # labels = [f"{predictors[0].make_label(param_name, value)}"
            #           for value in param_vals]
            labels = [f"{tex_map(value)}" for value in param_vals]
        else:
            raise ValueError

    else:
        raise ValueError

    for loss_plt, label in zip(loss, labels):
        plt_data = ax.plot(n_t_iter, loss_plt, label=label, marker=".")
        out.append(plt_data)

    if labels != [None]:
        ax.legend()

    ax.set(xlabel=r"$|\mathcal{T}|$")
    ax.set_title(title)

    return out
