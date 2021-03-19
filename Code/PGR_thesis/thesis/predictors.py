"""
Supervised learning functions.
"""

import math
from abc import ABC, abstractmethod
from itertools import product
from numbers import Integral
from typing import Union

# from more_itertools import all_equal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

from thesis.bayes import models as bayes_models
from thesis.loss_funcs import loss_se, loss_01
from thesis.random import elements as rand_elements, models as rand_models
from thesis.util import spaces
from thesis.util.base import vectorize_func, check_data_shape, all_equal
from thesis.util.spaces import check_spaces_x


def predict_stats_compare(predictors, model, params=None, x=None, n_train=0, n_mc=1, stats=('mode',), verbose=False):

    # TODO: Welford's online algorithm for mean and var calculation

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
            y_stats['mode'] = mode(y, axis=0)

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


def plot_predict_stats_compare(predictors, model, params=None, x=None, n_train=0, n_mc=1, do_std=False, verbose=False,
                               ax=None):
    space_x = check_spaces_x(predictors)
    if x is None:
        x = space_x.x_plt

    stats = ('mean', 'std') if do_std else ('mean',)  # TODO: generalize for mode, etc.
    y_stats_full = predict_stats_compare(predictors, model, params, x, n_train, n_mc, stats, verbose)

    if ax is None:
        ax = space_x.make_axes()

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

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
                    labels = [f"{predictor.tex_params(param_name, val)}" for val in param_vals]
            elif len(param_vals) == 1:
                y_stats = y_stats.squeeze(axis=1)
                labels = [f"$N = {n}$" for n in n_train]
                # title += f", ${param_name} = {param_vals[0]}$"
                title += f", {predictor.tex_params(param_name, param_vals[0])}"
            else:
                raise ValueError
        else:
            raise ValueError

        for y_stat, label in zip(y_stats, labels):
            y_mean = y_stat['mean']
            y_std = y_stat['std'] if do_std else None
            plt_data = space_x.plot_xy(x, y_mean, y_std, ax=ax, label=label)
            out.append(plt_data)

        if labels != [None]:
            ax.legend()

    else:
        if len(n_train) == 1:
            # TODO: enumerates and kwargs for errorbar predict. keep?
            # lens = [1 if len(p) == 0 else len(list(p.values())[0]) for p in params_full]
            # n_lines = sum(lens)

            title = f'$N = {n_train[0]}$'
            for predictor, params, y_stats in zip(predictors, params_full, y_stats_full):
                if len(params) == 0:
                    labels = [predictor.name]
                elif len(params) == 1:
                    y_stats = y_stats.squeeze(0)
                    param_name, param_vals = list(params.items())[0]
                    # labels = [f"{predictor.name}, ${param_name} = {val}$" for val in param_vals]
                    labels = [f"{predictor.name}, {predictor.tex_params(param_name, val)}" for val in param_vals]
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
                    y_mean = y_stat['mean']
                    y_std = y_stat['std'] if do_std else None
                    plt_data = space_x.plot_xy(x, y_mean, y_std, ax=ax, label=label)
                    out.append(plt_data)
        else:
            raise ValueError("Plotting not supported for multiple predictors and multiple values of n_train.")

        ax.legend()

    ax.set_title(title)

    return out


def risk_eval_sim_compare(predictors, model, params=None, n_train=0, n_test=1, n_mc=1, verbose=False):
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    n_train_delta = np.diff(np.concatenate(([0], list(n_train))))

    loss_full = []
    for params in params_full:
        params_shape = tuple(len(vals) for _, vals in params.items())
        # loss = np.empty((n_mc, len(n_train_delta), *params_shape))
        loss = np.zeros((len(n_train_delta), *params_shape))
        loss_full.append(loss)

    for i_mc in range(n_mc):
        if verbose:
            print(f"Loss iteration: {i_mc + 1}/{n_mc}")

        d = model.rvs(n_test + n_train_delta.sum())
        d_test, _d_train = d[:n_test], d[n_test:]
        d_train_iter = np.split(_d_train, np.cumsum(n_train_delta)[:-1])

        for i_n, d_train in enumerate(d_train_iter):
            warm_start = i_n > 0  # resets learner for new iteration
            for predictor, params, loss in zip(predictors, params_full, loss_full):
                predictor.fit(d_train, warm_start=warm_start)

                if len(params) == 0:
                    # loss[i_mc, i_n] = predictor.evaluate(d_test)
                    loss[i_n] += predictor.evaluate(d_test)
                else:
                    for i_v, param_vals in enumerate(list(product(*params.values()))):
                        predictor.set_params(**dict(zip(params.keys(), param_vals)))
                        # loss[i_mc, i_n][np.unravel_index(i_v, loss.shape[2:])] = predictor.evaluate(d_test)
                        loss[i_n][np.unravel_index(i_v, loss.shape[1:])] += predictor.evaluate(d_test)

    # loss_full = [loss.mean() for loss in loss_full]
    loss_full = [loss / n_mc for loss in loss_full]
    return loss_full


def risk_eval_comp_compare(predictors, model, params=None, n_train=0, n_test=1, verbose=False):
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
                idx = (np.arange(len(n_train)),) + tuple([k for _ in range(len(n_train))] for k in idx_p)
                loss[idx] = predictor.evaluate_comp(model, n_train, n_test)
                # loss[:, np.unravel_index(i_v, loss.shape[2:])] = predictor.evaluate_comp(model, n_train)

            loss_full.append(loss)

    return loss_full


def _plot_risk_eval_compare(losses, do_bayes, predictors, params=None, n_train=0, ax=None):
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    if ax is None:
        _, ax = plt.subplots()
        if do_bayes:
            ylabel = r'$\mathcal{R}(f)$'
        else:
            ylabel = r'$\mathcal{R}_{\Theta}(f;\theta)$'
        ax.set(ylabel=ylabel)

    out = []
    if len(predictors) == 1:
        predictor, params, loss = predictors[0], params_full[0], losses[0]
        title = str(predictor.name)

        if len(params) == 0:
            loss = loss[np.newaxis]
            xlabel, x_plt = '$N$', n_train
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
                xlabel, x_plt = '$N$', n_train
                if len(param_vals) == 1:
                    # title += f", {param_name} = {param_vals[0]}"
                    title += f", {predictor.tex_params(param_name, param_vals[0])}"
                    labels = [None]
                else:
                    # labels = [f"{param_name} = {val}" for val in param_vals]
                    labels = [f"{predictor.tex_params(param_name, val)}" for val in param_vals]
        else:
            raise NotImplementedError("Only up to one varying parameter currently supported.")

        for loss_plt, label in zip(loss, labels):
            plt_data = ax.plot(x_plt, loss_plt, label=label)
            out.append(plt_data)

        if labels != [None]:
            ax.legend()
    else:
        if all_equal(params.keys() for params in params_full) and len(n_train) == 1 and len(params_full[0]) == 1:
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
            title = ''
            xlabel, x_plt = '$N$', n_train
            for predictor, params, loss in zip(predictors, params_full, losses):
                if len(params) == 0:
                    loss = loss[np.newaxis]
                    labels = [predictor.name]
                elif len(params) == 1:
                    loss = np.transpose(loss)
                    param_name, param_vals = list(params.items())[0]
                    # labels = [f"{predictor.name}, {param_name} = {val}" for val in param_vals]
                    labels = [f"{predictor.name}, {predictor.tex_params(param_name, val)}" for val in param_vals]
                else:
                    raise NotImplementedError("Only up to one varying parameter currently supported.")

                for loss_plt, label in zip(loss, labels):
                    plt_data = ax.plot(x_plt, loss_plt, label=label)
                    out.append(plt_data)

                ax.legend()

    ax.set(xlabel=xlabel)
    ax.set_title(title)

    return out


def plot_risk_eval_sim_compare(predictors, model, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, ax=None):
    losses = risk_eval_sim_compare(predictors, model, params, n_train, n_test, n_mc, verbose)
    do_bayes = isinstance(model, bayes_models.Base)
    return _plot_risk_eval_compare(losses, do_bayes, predictors, params, n_train, ax)


def plot_risk_eval_comp_compare(predictors, model, params=None, n_train=0, n_test=1, verbose=False, ax=None):
    losses = risk_eval_comp_compare(predictors, model, params, n_train, n_test, verbose)
    do_bayes = isinstance(model, bayes_models.Base)
    return _plot_risk_eval_compare(losses, do_bayes, predictors, params, n_train, ax)


def plot_risk_disc(predictors, model, params=None, n_train=0, n_test=1, n_mc=500, verbose=True, ax=None):
    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if not all_equal(params_full):
        raise ValueError
    # TODO: check models for equality

    losses = risk_eval_sim_compare(predictors, model, params, n_train, n_test, n_mc, verbose)

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    if ax is None:
        _, ax = plt.subplots()
        if isinstance(model, bayes_models.Base):
            ylabel = r'$\mathcal{R}(f)$'
        else:
            ylabel = r'$\mathcal{R}_{\Theta}(f;\theta)$'
        ax.set(ylabel=ylabel)

    loss = np.stack(losses, axis=-1)
    params = params_full[0]

    x_plt = np.array([len(pr.model.space['x'].values) for pr in predictors])  # discretization set size
    # title = str(predictors[0].name)
    title = r'$\mathrm{Dir}$'

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
            labels = [f"{predictors[0].tex_params(param_name, val)}" for val in param_vals]
        else:
            raise ValueError

    else:
        raise ValueError

    for loss_plt, label in zip(loss, labels):
        plt_data = ax.plot(x_plt, loss_plt, label=label, marker='.')
        out.append(plt_data)

    if labels != [None]:
        ax.legend()

    ax.set(xlabel=r'$|\mathcal{T}|$')
    ax.set_title(title)

    return out


# %%
class Base(ABC):
    def __init__(self, loss_func, space=None, proc_funcs=(), name=None):
        self.loss_func = loss_func

        self._space = space

        self.proc_funcs = list(proc_funcs)
        self.name = name

        self.model = None

    # @property
    # def space(self):
    #     if self._space is None:
    #         self._space = self._model_obj.space
    #     return self._space
    space = property(lambda self: self._space)

    # space = property(lambda self: self._model_obj.space)

    shape = property(lambda self: {key: space.shape for key, space in self.space.items()})
    size = property(lambda self: {key: space.size for key, space in self.space.items()})
    ndim = property(lambda self: {key: space.ndim for key, space in self.space.items()})

    dtype = property(lambda self: {key: space.dtype for key, space in self.space.items()})

    # shape = property(lambda self: self.model.shape)
    # size = property(lambda self: self.model.size)
    # ndim = property(lambda self: self.model.ndim)

    @property
    @abstractmethod
    def _model_obj(self):
        raise NotImplementedError

    def set_params(self, **kwargs):  # TODO: improve? wrapper to ignore non-changing param set?
        for key, val in kwargs.items():
            setattr(self._model_obj, key, val)

    def tex_params(self, key, val=None):
        return self._model_obj.tex_params(key, val)

    # def get_params(self, *args):
    #     return {arg: getattr(self._model_obj, arg) for arg in args}

    def _proc_predictors(self, x):
        for func in self.proc_funcs:
            x = func(x)
        return x

    def _proc_data(self, d):
        x = self._proc_predictors(d['x'])
        dtype = [('x', d.dtype['x'].base, x.shape[1:]), ('y', d.dtype['y'].base, d.dtype['y'].shape)]
        return np.array(list(zip(x, d['y'])), dtype=dtype)

    def fit(self, d=None, warm_start=False):
        if d is None:
            d = np.array([], dtype=[(c, self.dtype[c], self.shape[c]) for c in 'xy'])

        d = self._proc_data(d)
        return self._fit(d, warm_start)

    @abstractmethod
    def _fit(self, d=None, warm_start=False):
        raise NotImplementedError

    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        d = model.rvs(n_train, rng=rng)  # generate train/test data
        self.fit(d, warm_start)  # train learner

    def predict(self, x):
        x = self._proc_predictors(x)
        return self._predict(x)

    def _predict(self, x):
        return vectorize_func(self._predict_single, shape=self.shape['x'])(x)

    def _predict_single(self, x):
        # raise NotImplementedError("Method must be overwritten.")  # TODO: numeric approx with loss and predictive!?
        pass

    def evaluate(self, d):
        loss = self.loss_func(self.predict(d['x']), d['y'], shape=self.shape['y'])
        return loss.mean()

    def evaluate_from_model(self, model, n_test=1, n_mc=1, rng=None):
        """Average empirical risk achieved from a given data model."""
        model.rng = rng
        loss = np.empty(n_mc)
        for i_mc in range(n_mc):
            d = model.rvs(n_test)  # generate train/test data
            loss[i_mc] = self.evaluate(d)  # make decision and assess

        return loss.mean()

    # Plotting utilities
    def plot_predict(self, x=None, ax=None, label=None):
        """Plot prediction function."""
        return self.space['x'].plot(self.predict, x, ax=ax, label=label)

    # Prediction statistics
    def predict_stats(self, model=None, params=None, x=None, n_train=0, n_mc=1, stats=('mode',), verbose=False):
        if model is None:
            model = self._model_obj
        return predict_stats_compare([self], model, [params], x, n_train, n_mc, stats, verbose)[0]

    def plot_predict_stats(self, model=None, params=None, x=None, n_train=0, n_mc=1, do_std=False, verbose=False,
                           ax=None):
        if model is None:
            model = self._model_obj
        return plot_predict_stats_compare([self], model, [params], x, n_train, n_mc, do_std, verbose, ax)

    # Risk evaluation
    def risk_eval_sim(self, model=None, params=None, n_train=0, n_test=1, n_mc=1, verbose=False):
        if model is None:
            model = self._model_obj
        return risk_eval_sim_compare([self], model, [params], n_train, n_test, n_mc, verbose)[0]

    def plot_risk_eval_sim(self, model=None, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, ax=None):
        if model is None:
            model = self._model_obj
        return plot_risk_eval_sim_compare([self], model, [params], n_train, n_test, n_mc, verbose, ax)

    def risk_eval_comp(self, model=None, params=None, n_train=0, n_test=1, verbose=False):
        if model is None:
            model = self._model_obj
        return risk_eval_comp_compare([self], model, [params], n_train, n_test, verbose)[0]

    def plot_risk_eval_comp(self, model=None, params=None, n_train=0, n_test=1, verbose=False, ax=None):
        if model is None:
            model = self._model_obj
        return plot_risk_eval_comp_compare([self], model, [params], n_train, n_test, verbose, ax)


class ClassifierMixin:
    model: rand_models.Base

    def _predict(self, x):
        return self.model.mode_y_x(x)


class RegressorMixin:
    model: Union[rand_models.Base, rand_models.MixinRVy]

    def _predict(self, x):
        return self.model.mean_y_x(x)


# %% Fixed model
class Model(Base):
    def __init__(self, model, loss_func, space=None, proc_funcs=(), name=None):
        if space is None:
            space = model.space
        super().__init__(loss_func, space, proc_funcs, name)
        self.model = model

    def __repr__(self):
        return self.__class__.__name__ + f"(model={self.model})"

    @property
    def _model_obj(self):
        return self.model

    def _fit(self, d=None, warm_start=False):
        pass

    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        pass  # skip unnecessary data generation


class ModelClassifier(ClassifierMixin, Model):
    def __init__(self, model, space=None, proc_funcs=(), name=None):
        super().__init__(model, loss_01, space, proc_funcs, name)


class ModelRegressor(RegressorMixin, Model):
    def __init__(self, model, space=None, proc_funcs=(), name=None):
        super().__init__(model, loss_se, space, proc_funcs, name)

    def evaluate_comp(self, model=None, n_train=0, n_test=1):
        if model is None:
            model = self._model_obj

        n_train = np.array(n_train)

        if isinstance(model, (rand_models.Base, rand_models.MixinRVy)):
            if isinstance(model.space['x'], spaces.FiniteGeneric):
                x = model.space['x'].values_flat

                p_x = model.model_x.pf(x)

                cov_y_x = model.cov_y_x(x)
                bias_sq = (self.predict(x) - model.mean_y_x(x)) ** 2

                risk = np.dot(cov_y_x + bias_sq, p_x)
                return np.full(n_train.shape, risk)
            else:
                raise NotImplementedError

        elif isinstance(model, bayes_models.Base):
            raise NotImplementedError


# %% Bayes model

class Bayes(Base):
    def __init__(self, bayes_model, loss_func, space=None, proc_funcs=(), name=None):
        if space is None:
            space = bayes_model.space
        super().__init__(loss_func, space, proc_funcs, name=name)

        self.bayes_model = bayes_model

        self.prior = self.bayes_model.prior
        self.posterior = self.bayes_model.posterior

        self.model = self.bayes_model.posterior_model  # updates in-place with set_params() and fit()

        self.fit()

    def __repr__(self):
        return self.__class__.__name__ + f"(bayes_model={self.bayes_model})"

    @property
    def _model_obj(self):
        return self.bayes_model

    def _fit(self, d=None, warm_start=False):
        self.bayes_model.fit(d, warm_start)

    def plot_param_dist(self, x=None, ax_prior=None):  # TODO: improve or remove?
        if x is None:
            raise ValueError  # TODO

        self.prior.plot_pf(x, ax=ax_prior)
        # ax_posterior= plt_prior.axes
        # ax_posterior = plt.gca()
        ax_posterior = None
        self.posterior.plot_pf(x, ax=ax_posterior)


class BayesClassifier(ClassifierMixin, Bayes):
    def __init__(self, bayes_model, space=None, proc_funcs=(), name=None):
        super().__init__(bayes_model, loss_01, space, proc_funcs, name)


class BayesRegressor(RegressorMixin, Bayes):
    def __init__(self, bayes_model, space=None, proc_funcs=(), name=None):
        super().__init__(bayes_model, loss_se, space, proc_funcs, name)

    def evaluate_comp(self, model=None, n_train=0, n_test=1):
        if model is None:
            model = self._model_obj

        n_train = np.array(n_train)

        if isinstance(model, (rand_models.Base, rand_models.MixinRVy)):
            if (isinstance(model.space['x'], spaces.FiniteGeneric)
                    and isinstance(self.bayes_model, bayes_models.Dirichlet)):

                x = model.space['x'].values_flat

                p_x = model.model_x.pf(x)
                alpha_x = self.bayes_model.alpha_0 * self.bayes_model.prior_mean.model_x.pf(x)

                cov_y_x = model.cov_y_x(x)
                bias_sq = (self.bayes_model.prior_mean.mean_y_x(x) - model.mean_y_x(x)) ** 2

                w_cov = np.zeros((n_train.size, p_x.size))
                w_bias = np.zeros((n_train.size, p_x.size))
                for i_n, n_i in enumerate(n_train.flatten()):
                    # rv = rand_elements.EmpiricalScalar(n_i, .5)
                    rv = rand_elements.Binomial(.5, n_i)
                    supp = rv.space.values
                    for i_x, (p_i, a_i) in enumerate(zip(p_x, alpha_x)):
                        rv.p = p_i
                        p_rv = rv.pf(supp)

                        # den = (a_i + n_i * supp) ** 2
                        den = (a_i + supp) ** 2

                        # w_cov[i_n, i_x] = (p_rv / den * n_i * supp).sum()
                        w_cov[i_n, i_x] = (p_rv / den * supp).sum()
                        w_bias[i_n, i_x] = (p_rv / den * a_i ** 2).sum()

                risk = np.dot(cov_y_x * (1 + w_cov) + bias_sq * w_bias, p_x)

                return risk.reshape(n_train.shape)
            else:
                raise NotImplementedError

        elif isinstance(model, bayes_models.Base):

            if (isinstance(model.space['x'], spaces.FiniteGeneric)
                    and isinstance(self.bayes_model, bayes_models.Dirichlet)):

                if (isinstance(model, bayes_models.Dirichlet) and model.alpha_0 == self.bayes_model.alpha_0
                        and model.prior_mean == self.bayes_model.prior_mean and n_test == 1):
                    # Minimum Bayesian squared-error

                    x = model.space['x'].values_flat

                    alpha_0 = self.bayes_model.alpha_0
                    alpha_m = self.bayes_model.prior_mean.model_x.pf(x)
                    weights = (alpha_m + 1 / (alpha_0 + n_train[..., np.newaxis])) / (alpha_m + 1 / alpha_0)

                    # return (alpha_m * weights * self.bayes_model.prior_mean.cov_y_x(x)).sum(axis=-1)
                    return np.dot(weights * self.bayes_model.prior_mean.cov_y_x(x), alpha_m)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
