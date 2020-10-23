"""
Supervised learning functions.
"""

import math
from numbers import Integral
from itertools import product
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

from util.generic import vectorize_func, check_data_shape
from util.plot import get_axes_xy
from loss_funcs import loss_se, loss_01
from models import Base as BaseModel, MixinRVy
from bayes_models import Base as BaseBayesModel


def predict_stats_compare(predictors, x, model, params=None, n_train=0, n_mc=1, stats=('mode',),
                          verbose=False, rng=None):

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = params

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    shape, size, ndim = model.shape, model.size, model.ndim
    x, set_shape = check_data_shape(x, shape['x'])
    n_train_delta = np.diff(np.concatenate(([0], list(n_train))))

    model = deepcopy(model)
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
            print(f"{i_mc+1}/{n_mc}")

        d = model.rvs(n_train_delta.sum())
        d_iter = np.split(d, np.cumsum(n_train_delta)[:-1])

        for i_n, d in enumerate(d_iter):
            warm_start = False if i_n == 0 else True  # resets learner for new iteration
            for predictor, params, params_shape, y in zip(predictors, params_full, params_shape_full, y_full):
                predictor.fit(d, warm_start=warm_start)
                if len(params) == 0:
                    y[i_mc, i_n] = predictor.predict(x)
                else:
                    for i_v, param_vals in enumerate(list(product(*params.values()))):
                        predictor.set_params(**dict(zip(params.keys(), param_vals)))
                        # params_shape = y.shape[2:-(len(set_shape) + ndim['y'])]
                        y[i_mc, i_n][np.unravel_index([i_v], params_shape)] = predictor.predict(x)

    # Generate statistics
    _samp, dtype = (), []
    for stat in stats:
        if stat in ('mode', 'median', 'mean'):
            stat_shape = set_shape + shape['y']
        elif stat in ('std', 'cov'):
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


def plot_predict_stats_compare(predictors, x, model, params=None, n_train=0, n_mc=1, do_std=False,
                               verbose=False, ax=None, rng=None):

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = params

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    stats = ('mean', 'std') if do_std else ('mean',)  # TODO: generalize for mode, etc.
    y_stats_full = predict_stats_compare(predictors, x, model, params_full, n_train, n_mc, stats, verbose, rng)

    ax = get_axes_xy(ax, model.shape['x'])

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
            plt_data = predictor.plot_xy(x, y_mean, y_std, ax, label=label)
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
                    plt_data = predictor.plot_xy(x, y_mean, y_std, ax, label=label)
                    out.append(plt_data)
        else:
            raise ValueError

        ax.legend()

    ax.set_title(title)

    return out


def loss_eval_compare(predictors, model, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, rng=None):

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = params

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    n_train_delta = np.diff(np.concatenate(([0], list(n_train))))

    model = deepcopy(model)
    model.rng = rng

    loss_full = []
    for params in params_full:
        params_shape = tuple(len(vals) for _, vals in params.items())
        loss = np.empty((n_mc, len(n_train_delta)) + params_shape)
        loss_full.append(loss)

    for i_mc in range(n_mc):
        if verbose:
            print(f"{i_mc+1}/{n_mc}")

        d = model.rvs(n_test + n_train_delta.sum())
        d_test, _d_train = d[:n_test], d[n_test:]
        d_train_iter = np.split(_d_train, np.cumsum(n_train_delta)[:-1])

        for i_n, d_train in enumerate(d_train_iter):
            warm_start = False if i_n == 0 else True  # resets learner for new iteration
            for predictor, params, loss in zip(predictors, params_full, loss_full):
                predictor.fit(d_train, warm_start=warm_start)

                if len(params) == 0:
                    loss[i_mc, i_n] = predictor.evaluate(d_test)
                else:
                    for i_v, param_vals in enumerate(list(product(*params.values()))):
                        predictor.set_params(**dict(zip(params.keys(), param_vals)))
                        loss[i_mc, i_n][np.unravel_index([i_v], loss.shape[2:])] = predictor.evaluate(d_test)

    loss_full = [loss.mean(axis=0) for loss in loss_full]
    return loss_full


def plot_loss_eval_compare(predictors, model, params=None, n_train=0, n_test=1, n_mc=1,
                           verbose=False, ax=None, rng=None):

    if params is None:
        params_full = [{} for _ in predictors]
    else:
        params_full = params

    if isinstance(n_train, (Integral, np.integer)):
        n_train = [n_train]

    loss_full = loss_eval_compare(predictors, model, params_full, n_train, n_test, n_mc, verbose, rng)

    if ax is None:
        _, ax = plt.subplots()
        ax.set(ylabel='Loss')
        ax.grid(True)

    out = []
    if len(predictors) == 1:
        predictor, params, loss = predictors[0], params_full[0], loss_full[0]
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
                    title += f"{param_name} = {param_vals[0]}"
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
        for predictor, params, loss in zip(predictors, params_full, loss_full):
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


# %%
class Base(ABC):
    def __init__(self, loss_func, name=None):
        self.loss_func = loss_func
        self.name = name

        self.model = None

    # def __repr__(self):
    #     return self.__class__.__name__ + f"(model={self.model})"

    shape = property(lambda self: self.model.shape)
    size = property(lambda self: self.model.size)
    ndim = property(lambda self: self.model.ndim)

    @property
    @abstractmethod
    def _model_obj(self):
        raise NotImplementedError

    def set_params(self, **kwargs):     # TODO: improve?
        for key, val in kwargs.items():
            setattr(self._model_obj, key, val)

    def get_params(self, *args):
        return {arg: getattr(self._model_obj, arg) for arg in args}

    @abstractmethod
    def fit(self, d=None, warm_start=False):
        raise NotImplementedError

    @abstractmethod
    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        raise NotImplementedError

    def predict(self, x):
        return vectorize_func(self._predict_single, data_shape=self.shape['x'])(x)

    def _predict_single(self, x):
        # raise NotImplementedError("Method must be overwritten.")  # TODO: numeric approx with loss and predictive!?
        pass

    def evaluate(self, d):
        loss = self.loss_func(self.predict(d['x']), d['y'], data_shape=self.shape['y'])
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
    def plot_xy(self, x, y, y_std=None, ax=None, label=None):
        # TODO: get 'x' default from model_x.plot_pf plot_data.axes?

        x, set_shape = check_data_shape(x, self.shape['x'])

        ax = get_axes_xy(ax, self.shape['x'])
        if self.ndim['y'] == 0:
            if self.shape['x'] == () and len(set_shape) == 1:
                plt_data = ax.plot(x, y, label=label)
                if y_std is not None:
                    # plt_data_std = ax.errorbar(x, y_mean, yerr=y_std)
                    plt_data_std = ax.fill_between(x, y - y_std, y + y_std, alpha=0.5)
                    plt_data = (plt_data, plt_data_std)

            elif self.shape['x'] == (2,) and len(set_shape) == 2:
                plt_data = ax.plot_surface(x[..., 0], x[..., 1], y, cmap=plt.cm.viridis)
                if y_std is not None:
                    plt_data_lo = ax.plot_surface(x[..., 0], x[..., 1], y - y_std, cmap=plt.cm.viridis)
                    plt_data_hi = ax.plot_surface(x[..., 0], x[..., 1], y + y_std, cmap=plt.cm.viridis)
                    plt_data = (plt_data, (plt_data_lo, plt_data_hi))

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return plt_data

    def plot_predict(self, x, ax=None, label=None):
        """Plot prediction function."""
        return self.plot_xy(x, self.predict(x), ax=ax, label=label)

    # # Prediction statistics
    # def predict_stats(self, x, model, params=None, n_train=0, n_mc=1, stats=('mode',), verbose=False, rng=None):
    #     if params is None:
    #         params = {}
    #     return predict_stats_compare([self], x, model, [params], n_train, n_mc, stats, verbose, rng)[0]
    #
    # def plot_predict_stats(self, x, model, params=None, n_train=0, n_mc=1, do_std=False,
    #                        verbose=False, ax=None, rng=None):
    #     if params is None:
    #         params = {}
    #     return plot_predict_stats_compare([self], x, model, [params], n_train, n_mc, do_std, verbose, ax, rng)
    #
    # # Loss evaluation
    # def loss_eval(self, model, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, rng=None):
    #     if params is None:
    #         params = {}
    #     return loss_eval_compare([self], model, [params], n_train, n_test, n_mc, verbose, rng)[0]
    #
    # def plot_loss_eval(self, model, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, ax=None, rng=None):
    #     if params is None:
    #         params = {}
    #     return plot_loss_eval_compare([self], model, [params], n_train, n_test, n_mc, verbose, ax, rng)

    # Prediction statistics
    def predict_stats(self, x, model=None, params=None, n_train=0, n_mc=1, stats=('mode',), verbose=False, rng=None):
        if model is None:
            model = self._model_obj
        if params is None:
            params = {}
        return predict_stats_compare([self], x, model, [params], n_train, n_mc, stats, verbose, rng)[0]

    def plot_predict_stats(self, x, model=None, params=None, n_train=0, n_mc=1, do_std=False,
                           verbose=False, ax=None, rng=None):
        if model is None:
            model = self._model_obj
        if params is None:
            params = {}
        return plot_predict_stats_compare([self], x, model, [params], n_train, n_mc, do_std, verbose, ax, rng)

    # Loss evaluation
    def loss_eval(self, model=None, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, rng=None):
        if model is None:
            model = self._model_obj
        if params is None:
            params = {}
        return loss_eval_compare([self], model, [params], n_train, n_test, n_mc, verbose, rng)[0]

    def plot_loss_eval(self, model=None, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, ax=None, rng=None):
        if model is None:
            model = self._model_obj
        if params is None:
            params = {}
        return plot_loss_eval_compare([self], model, [params], n_train, n_test, n_mc, verbose, ax, rng)


class ClassifierMixin:
    model: BaseModel

    def predict(self, x):
        return self.model.mode_y_x(x)  # TODO: argmax?


class RegressorMixin:
    model: MixinRVy

    def predict(self, x):
        return self.model.mean_y_x(x)  # TODO: m1?


#%% Fixed model
class Model(Base):
    def __init__(self, model, loss_func, name=None):
        super().__init__(loss_func, name)
        self.model = model

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


#%% Bayes model

class Bayes(Base):
    def __init__(self, bayes_model, loss_func, name=None):
        super().__init__(loss_func, name=name)

        self.bayes_model = bayes_model

        self.prior = self.bayes_model.prior
        self.posterior = self.bayes_model.posterior

        self.model = self.bayes_model.posterior_model

        self.fit()

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
            x = self.prior.x_default

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


# %%

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
#         self._model_gen = functools.partial(DataConditional.finite_model, supp_x=supp_x['x'], supp_y=supp_y['y'], rng=None)
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
