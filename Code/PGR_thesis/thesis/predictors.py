"""
Supervised learning functions.
"""

import math
from numbers import Integral
from itertools import product
import copy
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

from thesis.util import spaces
from thesis.util.base import vectorize_func, check_data_shape
from thesis.loss_funcs import loss_se, loss_01

from thesis.random import elements as rand_elements, models as rand_models
from thesis.bayes import models as bayes_models
from thesis.util.spaces import check_spaces


def predict_stats_compare(predictors, model, params=None, x=None, n_train=0, n_mc=1, stats=('mode',), verbose=False,
                          rng=None):

    space_x = check_spaces([pr.model.model_x for pr in predictors])
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

    model = copy.deepcopy(model)
    model.rng = rng

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
            print(f"Stats iteration: {i_mc+1}/{n_mc}")

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
                        # y[i_mc, i_n][np.unravel_index([i_v], params_shape)] = predictor.predict(x)
                        y[i_mc, i_n][np.unravel_index(i_v, params_shape)] = predictor.predict(x)

    # Generate statistics
    _samp, dtype = (), []
    for stat in stats:
        if stat in {'mode', 'median', 'mean'}:
            stat_shape = set_shape + shape['y']
        elif stat in {'std', 'cov'}:
            stat_shape = set_shape + 2 * shape['y']
        else:
            raise ValueError
        _samp += (np.empty(stat_shape),)
        dtype.append((stat, np.float, stat_shape))  # TODO: dtype float? need model dtype attribute?!

    y_stats_full = [np.tile(np.array(_samp, dtype=dtype), reps=(len(n_train_delta),) + param_shape)
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
                               ax=None, rng=None):

    stats = ('mean', 'std') if do_std else ('mean',)  # TODO: generalize for mode, etc.
    y_stats_full = predict_stats_compare(predictors, model, params, x, n_train, n_mc, stats, verbose, rng)

    space_x = check_spaces([pr.model.model_x for pr in predictors])

    if x is None:
        x = space_x.x_plt
    if ax is None:
        ax = space_x.make_axes(grid=True)

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
                title += f", N = {n_train[0]}"
            else:
                labels = [f"N = {n}" for n in n_train]
        elif len(params) == 1:
            param_name, param_vals = list(params.items())[0]
            if len(n_train) == 1:
                y_stats = y_stats.squeeze(axis=0)
                title += f", N = {n_train[0]}"
                if len(param_vals) == 1:
                    labels = [None]
                    title += f", {param_name} = {param_vals[0]}"
                else:
                    labels = [f"{param_name} = {val}" for val in param_vals]
            elif len(param_vals) == 1:
                y_stats = y_stats.squeeze(axis=1)
                labels = [f"N = {n}" for n in n_train]
                title += f", {param_name} = {param_vals[0]}"
            else:
                raise ValueError
        else:
            raise ValueError

        for y_stat, label in zip(y_stats, labels):
            y_mean = y_stat['mean']
            y_std = y_stat['std'] if do_std else None
            # plt_data = predictor.plot_xy(x, y_mean, y_std, ax, label=label)
            # plt_data = plot_xy(x, y_mean, y_std, space_x, ax, label=label)
            plt_data = space_x.plot_xy(x, y_mean, y_std, ax, label=label)
            out.append(plt_data)

        if labels != [None]:
            ax.legend()

    else:
        if len(n_train) == 1:
            title = f'N = {n_train[0]}'
            for predictor, params, y_stats in zip(predictors, params_full, y_stats_full):
                if len(params) == 0:
                    labels = [predictor.name]
                elif len(params) == 1:
                    y_stats = y_stats.squeeze(0)
                    param_name, param_vals = list(params.items())[0]
                    labels = [f"{predictor.name}, {param_name} = {val}" for val in param_vals]
                else:
                    raise ValueError

                for y_stat, label in zip(y_stats, labels):
                    y_mean = y_stat['mean']
                    y_std = y_stat['std'] if do_std else None
                    # plt_data = predictor.plot_xy(x, y_mean, y_std, ax, label=label)
                    # plt_data = plot_xy(x, y_mean, y_std, space_x, ax, label=label)
                    plt_data = space_x.plot_xy(x, y_mean, y_std, ax, label=label)
                    out.append(plt_data)
        else:
            raise ValueError("Plotting not supported for multiple predictors and multiple values of n_train.")

        ax.legend()

    ax.set_title(title)

    return out


def risk_eval_sim_compare(predictors, model, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, rng=None):

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    n_train_delta = np.diff(np.concatenate(([0], list(n_train))))

    model = copy.deepcopy(model)
    model.rng = rng

    loss_full = []
    for params in params_full:
        params_shape = tuple(len(vals) for _, vals in params.items())
        loss = np.empty((n_mc, len(n_train_delta)) + params_shape)
        loss_full.append(loss)

    for i_mc in range(n_mc):
        if verbose:
            print(f"Loss iteration: {i_mc+1}/{n_mc}")

        d = model.rvs(n_test + n_train_delta.sum())
        d_test, _d_train = d[:n_test], d[n_test:]
        d_train_iter = np.split(_d_train, np.cumsum(n_train_delta)[:-1])

        for i_n, d_train in enumerate(d_train_iter):
            warm_start = i_n > 0  # resets learner for new iteration
            for predictor, params, loss in zip(predictors, params_full, loss_full):
                predictor.fit(d_train, warm_start=warm_start)

                if len(params) == 0:
                    loss[i_mc, i_n] = predictor.evaluate(d_test)
                else:
                    for i_v, param_vals in enumerate(list(product(*params.values()))):
                        predictor.set_params(**dict(zip(params.keys(), param_vals)))
                        # loss[i_mc, i_n][np.unravel_index([i_v], loss.shape[2:])] = predictor.evaluate(d_test)
                        loss[i_mc, i_n][np.unravel_index(i_v, loss.shape[2:])] = predictor.evaluate(d_test)

    loss_full = [loss.mean(axis=0) for loss in loss_full]
    return loss_full


def risk_eval_comp_compare(predictors, model, params=None, n_train=0, n_test=1, verbose=False):

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    model = copy.deepcopy(model)

    loss_full = []
    for predictor, params in zip(predictors, params_full):
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


def _plot_risk_eval_compare(losses, predictors, params=None, n_train=0, ax=None):

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = [item if item is not None else {} for item in params]

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    if ax is None:
        _, ax = plt.subplots()
        ax.set(ylabel='Loss')
        ax.grid(True)

    out = []
    if len(predictors) == 1:
        predictor, params, loss = predictors[0], params_full[0], losses[0]
        title = str(predictor.name)

        if len(params) == 0:
            loss = loss[np.newaxis]
            xlabel, x_plt = 'N', n_train
            labels = [None]
        elif len(params) == 1:
            param_name, param_vals = list(params.items())[0]
            if len(n_train) < len(param_vals):
                xlabel, x_plt = param_name, param_vals
                if len(n_train) == 1:
                    title += f", N = {n_train[0]}"
                    labels = [None]
                else:
                    labels = [f"N = {n}" for n in n_train]
            else:
                loss = np.transpose(loss)
                xlabel, x_plt = 'N', n_train
                if len(param_vals) == 1:
                    title += f", {param_name} = {param_vals[0]}"
                    labels = [None]
                else:
                    labels = [f"{param_name} = {val}" for val in param_vals]
        else:
            raise ValueError

        for loss_plt, label in zip(loss, labels):
            plt_data = ax.plot(x_plt, loss_plt, label=label)
            out.append(plt_data)

        if labels != [None]:
            ax.legend()
    else:
        title = ''
        xlabel, x_plt = 'N', n_train
        for predictor, params, loss in zip(predictors, params_full, losses):
            if len(params) == 0:
                loss = loss[np.newaxis]
                labels = [predictor.name]
            elif len(params) == 1:
                loss = np.transpose(loss)
                param_name, param_vals = list(params.items())[0]
                labels = [f"{predictor.name}, {param_name} = {val}" for val in param_vals]
            else:
                raise ValueError

            for loss_plt, label in zip(loss, labels):
                plt_data = ax.plot(x_plt, loss_plt, label=label)
                out.append(plt_data)

            ax.legend()

    ax.set(xlabel=xlabel)
    ax.set_title(title)

    return out


def plot_risk_eval_sim_compare(predictors, model, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, ax=None,
                               rng=None):
    losses = risk_eval_sim_compare(predictors, model, params, n_train, n_test, n_mc, verbose, rng)
    return _plot_risk_eval_compare(losses, predictors, params, n_train, ax)


def plot_risk_eval_comp_compare(predictors, model, params=None, n_train=0, n_test=1, verbose=False, ax=None):
    losses = risk_eval_comp_compare(predictors, model, params, n_train, n_test, verbose)
    return _plot_risk_eval_compare(losses, predictors, params, n_train, ax)


#%%
class Base(ABC):
    def __init__(self, loss_func, name=None):
        self.loss_func = loss_func
        self.name = name

        self.model = None

    space = property(lambda self: self._model_obj.space)

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

    def set_params(self, **kwargs):     # TODO: improve? wrapper to ignore non-changing param set?
        for key, val in kwargs.items():
            setattr(self._model_obj, key, val)

    # def get_params(self, *args):
    #     return {arg: getattr(self._model_obj, arg) for arg in args}

    @abstractmethod
    def fit(self, d=None, warm_start=False):
        raise NotImplementedError

    @abstractmethod
    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        raise NotImplementedError

    def predict(self, x):
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
    def predict_stats(self, model=None, params=None, x=None, n_train=0, n_mc=1, stats=('mode',), verbose=False,
                      rng=None):
        if model is None:
            model = self._model_obj
        return predict_stats_compare([self], model, [params], x, n_train, n_mc, stats, verbose, rng)[0]

    def plot_predict_stats(self, model=None, params=None, x=None, n_train=0, n_mc=1, do_std=False, verbose=False,
                           ax=None, rng=None):
        if model is None:
            model = self._model_obj
        return plot_predict_stats_compare([self], model, [params], x, n_train, n_mc, do_std, verbose, ax, rng)

    # Risk evaluation
    def risk_eval_sim(self, model=None, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, rng=None):
        if model is None:
            model = self._model_obj
        return risk_eval_sim_compare([self], model, [params], n_train, n_test, n_mc, verbose, rng)[0]

    def plot_risk_eval_sim(self, model=None, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, ax=None,
                           rng=None):
        if model is None:
            model = self._model_obj
        return plot_risk_eval_sim_compare([self], model, [params], n_train, n_test, n_mc, verbose, ax, rng)

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

    def predict(self, x):
        return self.model.mode_y_x(x)


class RegressorMixin:
    model: Union[rand_models.Base, rand_models.MixinRVy]

    def predict(self, x):
        return self.model.mean_y_x(x)


#%% Fixed model
class Model(Base):
    def __init__(self, model, loss_func, name=None):
        super().__init__(loss_func, name)
        self.model = model

    def __repr__(self):
        return self.__class__.__name__ + f"(model={self.model})"

    @property
    def _model_obj(self):
        return self.model

    def fit(self, d=None, warm_start=False):
        pass

    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        pass


class ModelClassifier(ClassifierMixin, Model):
    def __init__(self, model, name=None):
        super().__init__(model, loss_01, name)


class ModelRegressor(RegressorMixin, Model):
    def __init__(self, model, name=None):
        super().__init__(model, loss_se, name)

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


#%% Bayes model

class Bayes(Base):
    def __init__(self, bayes_model, loss_func, name=None):
        super().__init__(loss_func, name=name)

        self.bayes_model = bayes_model

        self.prior = self.bayes_model.prior
        self.posterior = self.bayes_model.posterior

        self.model = self.bayes_model.posterior_model       # updates in-place with set_params() and fit()

        self.fit()

    def __repr__(self):
        return self.__class__.__name__ + f"(bayes_model={self.bayes_model})"

    @property
    def _model_obj(self):
        return self.bayes_model

    def fit(self, d=None, warm_start=False):
        self.bayes_model.fit(d, warm_start)

    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        d = model.rvs(n_train, rng=rng)  # generate train/test data
        self.fit(d, warm_start)  # train learner

    def plot_param_dist(self, x=None, ax_prior=None):  # TODO: improve or remove?
        if x is None:
            raise ValueError        # TODO

        self.prior.plot_pf(x, ax=ax_prior)
        # ax_posterior= plt_prior.axes
        # ax_posterior = plt.gca()
        ax_posterior = None
        self.posterior.plot_pf(x, ax=ax_posterior)


class BayesClassifier(ClassifierMixin, Bayes):
    def __init__(self, bayes_model, name=None):
        super().__init__(bayes_model, loss_01, name)


class BayesRegressor(RegressorMixin, Bayes):
    def __init__(self, bayes_model, name=None):
        super().__init__(bayes_model, loss_se, name)

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
                    rv = rand_elements.Binomial(n_i, .5)
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
                        and model.prior_mean == self.bayes_model.prior_mean):
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



#%%

# class DirichletFiniteClassifier(BaseLearner):
#     def __init__(self, alpha_0, mean_y_x):
#         super().__init__()
#         self.loss_func = loss_01
#
#         self.alpha_0 = alpha_0
#         self.mean_y_x = mean_y_x
#
#     def fit(self, d):
#
#
#     def _predict_single(self, x):
#         pass


# class BetaEstimatorTemp(Bayes):
#     def __init__(self, n_x=10):
#         super().__init__()
#         self.loss_fcn = loss_se
#         self.n_x = n_x
#         self.avg_y_x = np.zeros(n_x)
#
#     def fit(self, d=None):
#         delta = 1 / self.n_x
#         for i in range(self.n_x):
#             flag_match = np.logical_and(d['x'] >= i * delta, d['x'] < (i + 1) * delta)
#             if flag_match.any():
#                 self.avg_y_x[i] = d[flag_match]['y'].mean()
#
#     def _predict_single(self, x):
#         i = floor(x * self.n_x)
#         return self.avg_y_x[i]


# class Bayes(BaseLearner):
#     def __init__(self, supp_x, supp_y, alpha_0, mean):
#         super().__init__()
#
#         self.supp_x = supp_x        # Assumed to be my SL structured arrays!
#         self.supp_y = supp_y
#
#         self._supp_shape_x = supp_x.shape
#         self._supp_shape_y = supp_y.shape
#         self.data_shape_x = supp_x.dtype['x'].shape
#         self.data_shape_y = supp_y.dtype['y'].shape
#
#         self.alpha_0 = alpha_0
#         self.mean = mean
#
#         self._mean_x = mean.reshape(self._supp_shape_x + (-1,)).sum(axis=-1)
#
#         def _mean_y_x(x):
#             _mean_flat = mean.reshape((-1,) + self._supp_shape_y)
#             _mean_slice = _mean_flat[np.all(x.flatten()
#                                      == self.supp_x['x'].reshape(self.supp_x.size, -1), axis=-1)].squeeze(axis=0)
#             mean_y = _mean_slice / _mean_slice.sum()
#             return mean_y
#
#         self._mean_y_x = _mean_y_x
#
#         self._model_gen = functools.partial(DataConditional.finite_model,
#                                             supp_x=supp_x['x'], supp_y=supp_y['y'], rng=None)
#         self._posterior_mean = None
#         self.fit()
#
#     @property
#     def mean_x(self):
#         return self._mean_x
#
#     @property
#     def posterior_model(self):
#         return self._posterior_mean
#
#     def fit(self, d=np.array([])):
#         n = len(d)
#
# if n == 0:
#     p_x, p_y_x = self._mean_x, self._mean_y_x
# else:
#
#     emp_dist_x = empirical_pmf(d['x'], self.supp_x['x'], self.data_shape_x)
#
#     def emp_dist_y_x(x):
#         d_match = d[np.all(x.flatten() == d['x'].reshape(n, -1), axis=-1)].squeeze()
#         if d_match.size == 0:
#             return np.empty(self._supp_shape_y)
#         return empirical_pmf(d_match['y'], self.supp_y['y'], self.data_shape_y)
#
#     c_prior_x = 1 / (1 + n / self.alpha_0)
#     p_x = c_prior_x * self._mean_x + (1 - c_prior_x) * emp_dist_x
#
#     def p_y_x(x):
#         i = (self.supp_x['x'].reshape(self._supp_shape_x + (-1,)) == x.flatten()).all(-1)
#         c_prior_y = 1 / (1 + (n * emp_dist_x[i]) / (self.alpha_0 * self._mean_x[i]))
#         return c_prior_y * self._mean_y_x(x) + (1 - c_prior_y) * emp_dist_y_x(x)
#
# self._posterior_mean = self._model_gen(p_x=p_x, p_y_x=p_y_x)
#
#     # @classmethod
#     # def prior_gen(cls, bayes_model):
#     #     return cls(bayes_model.supp_x, bayes_model.supp_y, bayes_model.prior.alpha_0, bayes_model.prior.mean)
#
#
# class ModelClassifier(Bayes):
#     def __init__(self, supp_x, supp_y, alpha_0, mean):
#         super().__init__(supp_x, supp_y, alpha_0, mean)
#         self.loss_func = loss_01
#
#     def _predict_single(self, x):
#         return self._posterior_mean.mode_y_x(x)
#
#
# class BayesEstimator(Bayes):
#     def __init__(self, supp_x, supp_y, alpha_0, mean):
#         super().__init__(supp_x, supp_y, alpha_0, mean)
#         self.loss_func = loss_se
#
#     def _predict_single(self, x):
#         return self._posterior_mean.mean_y_x(x)


def main():
    pass


if __name__ == '__main__':
    main()
