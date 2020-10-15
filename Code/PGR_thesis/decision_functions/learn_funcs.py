"""
Supervised learning functions.
"""

import functools
import math
from numbers import Integral
from collections import Iterable
from itertools import product


import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

import util
from util.generic import vectorize_func, empirical_pmf, vectorize_first_arg, check_data_shape
from loss_funcs import loss_se, loss_01
from models import Base as BaseModel, MixinRVy, DataConditional


# TODO: add method functionality to work with SKL, TF conventions?


# %%
class ClassifierMixin:
    model: BaseModel

    def predict(self, x):
        return self.model.mode_y_x(x)  # TODO: argmax?


class RegressorMixin:
    model: MixinRVy

    def predict(self, x):
        return self.model.mean_y_x(x)  # TODO: m1?


# %%
class ModelPredictor:
    def __init__(self, loss_func, model, name=None):
        self.loss_func = loss_func
        self.model = model

        if name is None:
            self.name = 'model'
        elif isinstance(name, str):
            self.name = name
        else:
            raise TypeError

    def __repr__(self):
        return self.__class__.__name__ + f"(model={self.model})"

    shape = property(lambda self: self.model.shape)
    size = property(lambda self: self.model.size)
    ndim = property(lambda self: self.model.ndim)

    def set_params(self, **kwargs):     # TODO: improve?
        for key, val in kwargs.items():
            setattr(self.model, key, val)

    def get_params(self, *args):
        return {arg: getattr(self.model, arg) for arg in args}

    # def __getattr__(self, item):
    #     if item in self.model.param_names:
    #         # return getattr(self.model, item)
    #         return self.model.__getattr(item)
    #     else:
    #         return super().__getattribute__(item)
    #         # return getattr(self, item)
    #         # raise AttributeError
    #
    # def __setattr__(self, key, val):
    #     if key in self.model.param_names:
    #         # self.__setattr__(key, val)
    #         # setattr(self.model, key, val)
    #         self.model.__setattr__(key, val)
    #         # return getattr(self.model, val)
    #     else:
    #         super().__setattr__(key, val)

    def predict(self, x):
        return vectorize_func(self._predict_single, data_shape=self.shape['x'])(x)

    def _predict_single(self, x):
        raise NotImplementedError("Method must be overwritten.")  # TODO: numeric approx with loss and predictive!?
        pass

    def fit(self, d=None, warm_start=False):
        pass

    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        pass

    def evaluate(self, d):
        loss = self.loss_func(self.predict(d['x']), d['y'], data_shape=self.shape['y'])
        return loss.mean()

    def evaluate_from_model(self, model, n_test=1, n_mc=1, rng=None):       # TODO: move MC looping elsewhere?
        """Average empirical risk achieved from a given data model."""

        model.rng = rng
        loss = np.empty(n_mc)
        for i_mc in range(n_mc):
            d = model.rvs(n_test)  # generate train/test data
            loss[i_mc] = self.evaluate(d)  # make decision and assess

        return loss.mean()

    # def evaluate_from_model(self, model, n_test=1, rng=None):
    #     """Average empirical risk achieved from a given data model."""
    #
    #     d = model.rvs(n_test, rng=rng)  # generate train/test data
    #     loss = self.evaluate(d)  # make decision and assess
    #
    #     return loss

    # Plotting utilities

    @staticmethod
    def _get_axes_xy(ax=None, shape=None):
        if ax is None:
            if shape == ():
                _, ax = plt.subplots()
                ax.set(xlabel='$x$', ylabel='$\\hat{y}(x)$')
            elif shape == (2,):
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$\\hat{y}(x)$')
            else:
                return None

            ax.grid(True)
            return ax
        else:
            return ax

    def plot_xy(self, x, y, y_std=None, ax=None, label=None):
        # TODO: get 'x' default from model_x.plot_pf plot_data.axes?

        x, set_shape = check_data_shape(x, self.shape['x'])

        ax = self._get_axes_xy(ax, self.shape['x'])
        if label is None:
            label = self.name

        if self.ndim['y'] == 0:
            if self.shape['x'] == () and len(set_shape) == 1:
                # if ax is None:
                #     _, ax = plt.subplots()
                #     ax.set(xlabel='$x$', ylabel='$\\hat{y}(x)$')
                #     ax.grid(True)

                plt_data = ax.plot(x, y, label=label)
                if y_std is not None:
                    # plt_data_std = ax.errorbar(x, y_mean, yerr=y_std)
                    plt_data_std = ax.fill_between(x, y - y_std, y + y_std, alpha=0.5)
                    plt_data = (plt_data, plt_data_std)

            elif self.shape['x'] == (2,) and len(set_shape) == 2:
                # if ax is None:
                #     _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                #     ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$\\hat{y}(x)$')

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

    # Prediction statistics

    # @staticmethod
    # def prediction_stats(predictors, x, model, n_train=(0,), n_mc=1, stats=('mode',), rng=None):
    #
    #     shape, size, ndim = model.shape, model.size, model.ndim
    #     if not all(predictor.shape == shape for predictor in predictors):
    #         raise ValueError("All models must have same shape.")
    #
    #     x, set_shape = check_data_shape(x, shape['x'])
    #     n_train_delta = np.diff(np.concatenate(([0], list(n_train))))
    #     model.rng = rng
    #
    #     # Generate random data and make predictions
    #     shape_out = (len(n_train_delta), len(predictors))
    #     y = np.empty((n_mc, *shape_out, *set_shape, *shape['y']))
    #     for i_mc in range(n_mc):
    #         for i_n, n_tr in enumerate(n_train_delta):
    #             warm_start = False if i_n == 0 else True  # resets learner for new iteration
    #             d = model.rvs(n_tr)
    #             for i_p, predictor in enumerate(predictors):
    #                 predictor.fit(d, warm_start=warm_start)
    #                 y[i_mc, i_n, i_p] = predictor.predict(x)
    #
    #     # Generate statistics
    #     _samp, dtype = (), []
    #     for stat in stats:
    #         if stat in ('mode', 'median', 'mean'):
    #             stat_shape = set_shape + shape['y']
    #         elif stat in ('std', 'cov'):
    #             stat_shape = set_shape + 2 * shape['y']
    #         else:
    #             raise ValueError
    #         _samp += (np.empty(stat_shape),)
    #         dtype.append((stat, np.float, stat_shape))  # TODO: dtype float? need model dtype attribute?!
    #
    #     y_stats = np.tile(np.array(_samp, dtype=dtype), reps=shape_out)
    #
    #     if 'mode' in stats:
    #         y_stats['mode'] = mode(y, axis=0)
    #
    #     if 'median' in stats:
    #         y_stats['median'] = np.median(y, axis=0)
    #
    #     if 'mean' in stats:
    #         y_stats['mean'] = y.mean(axis=0)
    #
    #     if 'std' in stats:
    #         if ndim['y'] == 0:
    #             y_stats['std'] = y.std(axis=0)
    #         else:
    #             raise ValueError("Standard deviation is only supported for singular data shapes.")
    #
    #     if 'cov' in stats:
    #         if size['y'] == 1:
    #             _temp = y.var(axis=0)
    #         else:
    #             _temp = np.moveaxis(y.reshape((n_mc, math.prod(set_shape), size['y'])), 0, -1)
    #             _temp = np.array([np.cov(t) for t in _temp])
    #
    #         y_stats['cov'] = _temp.reshape(set_shape + 2 * shape['y'])
    #
    #     return y_stats
    #
    # @classmethod
    # def plot_compare_stats(cls, predictors, x, model, n_train=0, n_mc=1, do_std=False, ax=None, rng=None):
    #
    #     if not isinstance(predictors, Iterable):
    #         predictors = [predictors]
    #
    #     if isinstance(n_train, (Integral, np.integer)):
    #         n_train = [n_train]
    #     stats = ('mean', 'std') if do_std else ('mean',)    # TODO: generalize for mode, etc.
    #
    #     y_stats = cls.prediction_stats(predictors, x, model, n_train, n_mc, stats, rng)
    #
    #     ax = cls._get_axes_xy(ax, model.shape['x'])
    #
    #     if len(n_train) == 1:
    #         p_iter = predictors
    #         y_stats = y_stats.squeeze(axis=0)
    #         labels = [p.name for p in predictors]
    #         ax.set_title(f'N = {n_train[0]}')
    #     else:
    #         if len(predictors) == 1:
    #             p_iter = len(n_train) * predictors
    #             y_stats = y_stats.squeeze(axis=1)
    #             labels = [f"N = {n}" for n in n_train]
    #             ax.set_title(f"{predictors[0].name}")
    #         else:
    #             raise ValueError
    #
    #     out = []
    #     for predictor, y_stat, label in zip(p_iter, y_stats, labels):
    #         y_mean = y_stat['mean']
    #         y_std = y_stat['std'] if do_std else None
    #         plt_data = predictor.plot_xy(x, y_mean, y_std, ax, label)
    #         out.append(plt_data)
    #
    #     ax.legend()
    #
    #     return out

    @staticmethod
    def prediction_stats(predictors, x, model, params=None, n_train=(0,), n_mc=1, stats=('mode',), rng=None):

        shape, size, ndim = model.shape, model.size, model.ndim
        if not all(predictor.shape == shape for predictor in predictors):
            raise ValueError("All models must have same shape.")

        x, set_shape = check_data_shape(x, shape['x'])
        n_train_delta = np.diff(np.concatenate(([0], list(n_train))))
        model.rng = rng

        if params is None:
            params_full = [{} for _ in predictors]
        else:
            params_full = params

        # Generate random data and make predictions

        # shape_out = (len(n_train_delta), len(predictors))
        # y = np.empty((n_mc, *shape_out, *set_shape, *shape['y']))
        # for i_mc in range(n_mc):
        #     for i_n, n_tr in enumerate(n_train_delta):
        #         warm_start = False if i_n == 0 else True  # resets learner for new iteration
        #         d = model.rvs(n_tr)
        #         for i_p, predictor in enumerate(predictors):
        #             predictor.fit(d, warm_start=warm_start)
        #             y[i_mc, i_n, i_p] = predictor.predict(x)

        y_full = []
        shape_out_full = []
        for predictor, params in zip(predictors, params_full):
            shape_out = (len(n_train_delta),) + tuple(len(vals) for _, vals in params.items())
            shape_out_full.append(shape_out)
            y = np.empty((n_mc, *shape_out, *set_shape, *shape['y']))
            y_full.append(y)

        for i_mc in range(n_mc):
            for i_n, n_tr in enumerate(n_train_delta):
                warm_start = False if i_n == 0 else True  # resets learner for new iteration
                d = model.rvs(n_tr)
                for i_p, (predictor, params) in enumerate(zip(predictors, params_full)):
                    predictor.fit(d, warm_start=warm_start)
                    for i_v, param_vals in enumerate(list(product(*params.values()))):
                        predictor.set_params(**dict(zip(params.keys(), param_vals)))
                        if len(params) == 0:
                            y_full[i_p][i_mc, i_n] = predictor.predict(x)
                        else:
                            y_full[i_p][i_mc, i_n][np.unravel_index([i_v],
                                                                    shape_out_full[i_p][1:])] = predictor.predict(x)

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

        # y_stats = np.tile(np.array(_samp, dtype=dtype), reps=shape_out)
        y_stats_full = [np.tile(np.array(_samp, dtype=dtype), reps=shape_out) for shape_out in shape_out_full]

        for y_stats, y in zip(y_stats_full, y_full):
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

    @classmethod
    def plot_compare_stats(cls, predictors, x, model, params=None, n_train=0, n_mc=1, do_std=False, ax=None, rng=None):

        # if isinstance(n_train, (Integral, np.integer)):
        #     n_train = [n_train]

        if params is None:
            params_full = [{} for _ in predictors]
        else:
            params_full = params

        stats = ('mean', 'std') if do_std else ('mean',)  # TODO: generalize for mode, etc.
        y_stats_full = cls.prediction_stats(predictors, x, model, params, [n_train], n_mc, stats, rng)

        ax = cls._get_axes_xy(ax, model.shape['x'])

        out = []
        for predictor, params, y_stats in zip(predictors, params_full, y_stats_full):
            if len(params) == 0:
                labels = [predictor.name]
            elif len(params) == 1:
                y_stats = y_stats.squeeze(0)
                name, vals = list(params.items())[0]
                labels = [f"{predictor.name}: {name} = {val}" for val in vals]
            else:
                raise ValueError

            for y_stat, label in zip(y_stats, labels):
                y_mean = y_stat['mean']
                y_std = y_stat['std'] if do_std else None
                plt_data = predictor.plot_xy(x, y_mean, y_std, ax, label=label)
                out.append(plt_data)

        ax.set_title(f'N = {n_train}')
        ax.legend()

        return out


    def prediction_stats_param(self, x, model, params=None, n_train=(0,), n_mc=1, stats=('mode',), rng=None):

        shape, size, ndim = model.shape, model.size, model.ndim
        if self.shape != shape:
            raise ValueError("Predictor and model must have the same shape.")

        x, set_shape = check_data_shape(x, shape['x'])
        n_train_delta = np.diff(np.concatenate(([0], list(n_train))))
        model.rng = rng

        if params is None:
            params = {}

        # Generate random data and make predictions
        shape_out = (len(n_train_delta),) + tuple(len(vals) for _, vals in params.items())
        y = np.empty((n_mc, *shape_out, *set_shape, *shape['y']))
        for i_mc in range(n_mc):
            for i_n, n_tr in enumerate(n_train_delta):
                warm_start = False if i_n == 0 else True  # resets learner for new iteration
                d = model.rvs(n_tr)
                self.fit(d, warm_start=warm_start)
                for i_v, param_vals in enumerate(list(product(*params.values()))):
                    self.set_params(**dict(zip(params.keys(), param_vals)))
                    if len(params) == 0:
                        y[i_mc, i_n] = self.predict(x)
                    else:
                        y[i_mc, i_n][np.unravel_index([i_v], shape_out[1:])] = self.predict(x)

        # Calculate statistics
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

        y_stats = np.tile(np.array(_samp, dtype=dtype), reps=shape_out)

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

        return y_stats

    def plot_predict_stats_param(self, x, model, params=None, n_train=0, n_mc=1, do_std=False, ax=None, rng=None):

        if isinstance(n_train, (Integral, np.integer)):
            n_train = [n_train]

        if params is None:
            params = {}

        ax = self._get_axes_xy(ax, model.shape['x'])

        # Generate statistics
        stats = ('mean', 'std') if do_std else ('mean',)  # TODO: generalize for mode, etc.
        y_stats = self.prediction_stats_param(x, model, params, n_train, n_mc, stats, rng)

        # Check plot type
        if len(params) == 0:
            labels = [f"N = {n}" for n in n_train]
            ax.set_title(f"{self.name}")
        elif len(params) == 1:
            name, vals = list(params.items())[0]
            if len(n_train) == 1:
                y_stats = y_stats.squeeze(axis=0)
                labels = [f"{name} = {val}" for val in vals]
                ax.set_title(f'N = {n_train[0]}')
            elif len(vals) == 1:
                y_stats = y_stats.squeeze(axis=1)
                labels = [f"N = {n}" for n in n_train]
                ax.set_title(f"{name} = {vals[0]}")
            else:
                raise ValueError
        else:
            raise ValueError

        # Plot
        out = []
        for y_stat, label in zip(y_stats, labels):
            y_mean = y_stat['mean']
            y_std = y_stat['std'] if do_std else None
            plt_data = self.plot_xy(x, y_mean, y_std, ax, label)
            out.append(plt_data)
        ax.legend()

        return out



    # Loss evaluation

    @staticmethod   # FIXME
    def compare_eval(predictors, model, n_train=(0,), n_test=1, n_mc=1, rng=None):
        # TODO: code redundancy?

        shape, size, ndim = model.shape, model.size, model.ndim
        if not all(predictor.shape == shape for predictor in predictors):
            raise ValueError("All models must have same shape.")

        n_train_delta = np.diff(np.concatenate(([0], list(n_train))))
        model.rng = rng

        # Generate random data and make predictions
        loss = np.empty((n_mc, len(n_train_delta), len(predictors)))
        for i_mc in range(n_mc):
            d_test = model.rvs(n_test)
            for i_n, n_tr in enumerate(n_train_delta):
                warm_start = False if i_n == 0 else True  # resets learner for new iteration
                d_train = model.rvs(n_tr)
                for i_p, predictor in enumerate(predictors):
                    predictor.fit(d_train, warm_start=warm_start)
                    loss[i_mc, i_n, i_p] = predictor.evaluate(d_test)

        return loss.mean(axis=0)

    @classmethod
    def plot_compare_eval(cls, predictors, model, n_train=0, n_test=1, n_mc=1, ax=None, rng=None):

        if not isinstance(predictors, Iterable):
            predictors = [predictors]

        if isinstance(n_train, (Integral, np.integer)):
            n_train = [n_train]

        if ax is None:
            _, ax = plt.subplots()
            ax.set(ylabel='Loss')
            ax.grid(True)

        loss = cls.compare_eval(predictors, model, n_train, n_test, n_mc, rng)

        if loss.shape[0] > loss.shape[1]:
            loss = np.transpose(loss)
            x_plt = n_train
            labels = [p.name for p in predictors]
            ax.set(xlabel='N')
        else:
            x_plt = range(len(predictors))
            labels = [f"N = {n}" for n in n_train]
            ax.set(xlabel='Predictors')     # TODO

        out = []
        for loss_plt, label in zip(loss, labels):
            plt_data = ax.plot(x_plt, loss_plt, label=label)
            out.append(plt_data)

        ax.legend()

        return out


class ModelClassifier(ClassifierMixin, ModelPredictor):
    def __init__(self, model, name=None):
        super().__init__(loss_01, model, name)


class ModelRegressor(RegressorMixin, ModelPredictor):
    def __init__(self, model, name=None):
        super().__init__(loss_se, model, name)


# %% Learning Functions

class BayesPredictor(ModelPredictor):
    def __init__(self, loss_func, bayes_model, name=None):
        super().__init__(loss_func, model=None, name=name)

        self.bayes_model = bayes_model

        self.fit()      # sets model attribute for prediction

    prior = property(lambda self: self.bayes_model.prior)
    posterior = property(lambda self: self.bayes_model.posterior)

    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self.bayes_model, key, val)

    def get_params(self, *args):
        return {arg: getattr(self.bayes_model, arg) for arg in args}

    def fit(self, d=None, warm_start=False):
        self.bayes_model.fit(d, warm_start)
        self.model = self.bayes_model.posterior_model

    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        d = model.rvs(n_train, rng=rng)  # generate train/test data
        self.fit(d, warm_start)  # train learner

    # def fiteval_from_model(self, model, n_train=0, n_test=1, n_mc=1, rng=None):
    #     """Average empirical risk achieved from a given data model."""
    #
    #     model.rng = rng
    #     loss = np.empty(n_mc)
    #     for i_mc in range(n_mc):
    #         d = model.rvs(n_train + n_test)  # generate train/test data
    #         d_train, d_test = np.split(d, [n_train])
    #
    #         self.fit(d_train)  # train learner
    #         loss[i_mc] = self.evaluate(d_test)  # make decision and assess
    #
    #     return loss.mean()

    def plot_param_dist(self, x=None, ax_prior=None):  # TODO: improve or remove?
        if x is None:
            x = self.prior.x_default

        self.prior.plot_pf(x, ax=ax_prior)
        # ax_posterior= plt_prior.axes
        # ax_posterior = plt.gca()
        ax_posterior = None
        self.posterior.plot_pf(x, ax=ax_posterior)

    # def prediction_stats_old(self, x, model, n_train=0, n_mc=1, stats=('mode',), rng=None):     # FIXME
    #     """Get mean and covariance of prediction function for a given data model."""
    #
    #     x, set_shape = check_data_shape(x, self.shape['x'])
    #
    #     if isinstance(n_train, (Integral, np.integer)):
    #         n_train = [n_train]
    #     n_train = np.array(n_train)
    #     n_train_delta = np.diff(np.concatenate(([0], n_train)))
    #
    #     model.rng = rng
    #     y_seq = np.empty((len(n_train), n_mc, *set_shape, *self.shape['y']))
    #     for i_mc in range(n_mc):
    #         for i_n, n_ in enumerate(n_train_delta):
    #             warm_start = False if i_n == 0 else True    # resets learner for new iteration
    #             self.fit_from_model(model, n_train=n_, warm_start=warm_start)
    #             y_seq[i_n, i_mc] = self.predict(x)
    #
    #     # y = np.empty((n_mc, *set_shape, *self.shape['y']))
    #     # for i_mc in range(n_mc):
    #     #     self.fit_from_model(model, n_train)
    #     #     y[i_mc] = self.predict(x)
    #
    #     out = []
    #     for y in y_seq:
    #         stats = {stat: None for stat in stats}
    #
    #         if 'mode' in stats.keys():
    #             stats['mode'] = mode(y, axis=0)
    #
    #         if 'median' in stats.keys():
    #             stats['median'] = np.median(y, axis=0)
    #
    #         if 'mean' in stats.keys():
    #             stats['mean'] = y.mean(axis=0)
    #
    #         if 'std' in stats.keys():
    #             if self.ndim['y'] == 0:
    #                 stats['std'] = y.std(axis=0)
    #             else:
    #                 raise ValueError("Standard deviation is only supported for singular data shapes.")
    #
    #         if 'cov' in stats.keys():
    #             if self.size['y'] == 1:
    #                 _temp = y.var(axis=0)
    #             else:
    #                 _temp = np.moveaxis(y.reshape((n_mc, math.prod(set_shape), self.size['y'])), 0, -1)
    #                 _temp = np.array([np.cov(t) for t in _temp])
    #
    #             stats['cov'] = _temp.reshape(set_shape + 2 * self.shape['y'])
    #
    #         out.append(stats)
    #
    #     # return stats
    #     return out


class BayesClassifier(ClassifierMixin, BayesPredictor):
    def __init__(self, bayes_model, name=None):
        super().__init__(loss_01, bayes_model, name)


class BayesRegressor(RegressorMixin, BayesPredictor):
    def __init__(self, bayes_model, name=None):
        super().__init__(loss_se, bayes_model, name)

        # FIXME
        # stat_str = ('mean', 'std') if do_std else ('mean',)
        # stats_seq = self.prediction_stats(x, model, n_train, n_mc, stats=stat_str, rng=rng)
        #
        # if len(stats_seq) == 1:
        #     labels = [self.name]
        # else:
        #     labels = [f"N = {n}" for n in n_train]
        #
        # for stats, label in zip(stats_seq, labels):
        #     y_mean = stats['mean']
        #
        #     plt_data = self.plot_xy(x, y_mean, ax, label=label)
        #
        #     if do_std:
        #         y_std = stats['std']
        #         ax = plt.gca()
        #         if self.shape['x'] == ():
        #             # plt_data_std = ax.errorbar(x, y_mean, yerr=y_std)
        #             plt_data_std = ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.5)
        #             plt_data = (plt_data, plt_data_std)
        #
        #         elif self.shape['x'] == (2,):
        #             plt_data_lo = ax.plot_surface(x[..., 0], x[..., 1], y_mean - y_std, cmap=plt.cm.viridis)
        #             plt_data_hi = ax.plot_surface(x[..., 0], x[..., 1], y_mean + y_std, cmap=plt.cm.viridis)
        #             plt_data = (plt_data, (plt_data_lo, plt_data_hi))
        #         else:
        #             raise NotImplementedError
        #     else:
        #         raise NotImplementedError

        # return plt_data


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


# class BetaEstimatorTemp(BayesPredictor):
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


# class BayesPredictor(BaseLearner):
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
# class ModelClassifier(BayesPredictor):
#     def __init__(self, supp_x, supp_y, alpha_0, mean):
#         super().__init__(supp_x, supp_y, alpha_0, mean)
#         self.loss_func = loss_01
#
#     def _predict_single(self, x):
#         return self._posterior_mean.mode_y_x(x)
#
#
# class BayesEstimator(BayesPredictor):
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
