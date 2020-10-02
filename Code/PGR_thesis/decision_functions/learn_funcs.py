"""
Supervised learning functions.
"""

import functools
from math import floor
from collections import Sequence

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

import util
from util.generic import vectorize_func, empirical_pmf, vectorize_first_arg, check_data_shape
from loss_funcs import loss_se, loss_01
from SL_obj import BaseModel, MixinRVy, YcXModel


# TODO: add method functionality to work with SKL, TF conventions?
# TODO: COMPLETE property set/get check, rework!


# TODO: infer learner type from loss_func? use same model attribute for both optimal and bayesian learners!


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

    # @property
    # def data_shape_x(self):
    #     return self.model.data_shape_x
    #
    # @property
    # def data_shape_y(self):
    #     return self.model.data_shape_y

    # def fit(self, d):
    #     pass

    def predict(self, x):
        return vectorize_func(self._predict_single, data_shape=self.shape['x'])(x)

    def _predict_single(self, x):
        raise NotImplementedError("Method must be overwritten.")  # TODO: numeric approx with loss and predictive!?
        pass

    def fit(self, d=None):
        pass

    def fit_from_model(self, model, n_train=0, rng=None):
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

    def plot_predict(self, x, ax=None):
        """Plot prediction function."""
        # TODO: get 'x' default from model_x.plot_pf plot_data.axes?

        if self.shape['x'] not in ((), (1,)):
            raise NotImplementedError

        if ax is None:
            _, ax = plt.subplots()
            ax.set(xlabel='$x$', ylabel='$\\hat{y}(x)$')
            ax.grid(True)

        plt_data = ax.plot(x, self.predict(x), label=self.name)

        return plt_data

    @classmethod
    def plot_predictions(cls, predictors, x, ax=None):  # TODO: improve or remove?
        """Plot multiple prediction functions on a single axes."""

        if ax is None:
            _, ax = plt.subplots()
            ax.grid(True)

        plt_data = []
        for predictor in predictors:
            plt_data_ = predictor.plot_predict(x, ax)
            plt_data.append(plt_data_[0])
        return plt_data


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

        self.prior = self.bayes_model.prior
        self.posterior = None

        self.fit()

    # @property
    # def data_shape_x(self):
    #     return self.bayes_model.data_shape_x
    #
    # @property
    # def data_shape_y(self):
    #     return self.bayes_model.data_shape_y

    shape = property(lambda self: self.bayes_model.shape)
    size = property(lambda self: self.bayes_model.size)
    ndim = property(lambda self: self.bayes_model.ndim)

    def fit(self, d=None):
        if d is None:
            d = np.array([], dtype=[('x', '<f8', self.shape['x']),
                                    ('y', '<f8', self.shape['y'])])

        self.posterior, self.model = self.bayes_model.fit(d)

    def fit_from_model(self, model, n_train=0, rng=None):
        d = model.rvs(n_train, rng=rng)  # generate train/test data
        self.fit(d)  # train learner

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

    def plot_param_dist(self, ax_prior=None):  # TODO: improve or remove?
        plt_prior = self.prior.plot_pf(ax=ax_prior)
        ax_prior = plt_prior.axes
        ax_posterior = ax_prior
        self.posterior.plot_pf(ax=ax_posterior)

    def prediction_stats(self, x, model, n_train=0, n_mc=1, stats=('mode',), rng=None):
        """Get mean and covariance of prediction function for a given data model."""

        x, set_shape = check_data_shape(x, self.shape['x'])

        model.rng = rng
        y = np.empty((n_mc, *set_shape, *self.shape['y']))
        for i_mc in range(n_mc):
            self.fit_from_model(model, n_train)
            # self.fit(model.rvs(n_train))
            y[i_mc] = self.predict(x)

        stats = {stat: None for stat in stats}

        if 'mode' in stats.keys():
            stats['mode'] = mode(y, axis=0)

        if 'median' in stats.keys():
            stats['median'] = np.median(y, axis=0)

        if 'mean' in stats.keys():
            stats['mean'] = y.mean(0)

        if 'cov' in stats.keys() or 'std' in stats.keys():
            try:
                y_mean = stats['mean']
            except KeyError:
                y_mean = y.mean(0)

            y_del = y - y_mean
            y_1 = y_del.reshape(n_mc, *set_shape, *self.shape['y'], *(1 for _ in self.shape['y']))
            y_2 = y_del.reshape(n_mc, *set_shape, *(1 for _ in self.shape['y']), *self.shape['y'])
            # y_cov = (y_1 * y_2).mean(0).reshape(*set_shape, *2 * self.shape['y])  # biased estimate
            y_cov = (y_1 * y_2).mean(0)  # biased estimate

            if 'cov' in stats.keys():
                stats['cov'] = y_cov
            if 'std' in stats.keys():
                stats['std'] = np.sqrt(y_cov)

        return stats


class BayesClassifier(ClassifierMixin, BayesPredictor):
    def __init__(self, bayes_model, name=None):
        super().__init__(loss_01, bayes_model, name)


class BayesRegressor(RegressorMixin, BayesPredictor):
    def __init__(self, bayes_model, name=None):
        super().__init__(loss_se, bayes_model, name)

    def plot_predict_stats(self, x, model, n_train=0, n_mc=1, do_std=False, ax=None, rng=None):
        stats = ('mean', 'std') if do_std else ('mean',)
        stats = self.prediction_stats(x, model, n_train, n_mc, stats=stats, rng=rng)
        y_mean = stats['mean']
        if do_std:
            y_std = stats['std']

        if self.shape['y'] == ():
            if self.shape['x'] == ():
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x$', ylabel='$\\hat{y}(x)$')
                    ax.grid(True)

                plt_data = ax.plot(x, y_mean)
                if do_std:
                    # plt_data_std = ax.errorbar(x, y_mean, yerr=y_std)
                    plt_data_std = ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.5)
                    plt_data = (plt_data, plt_data_std)

            elif self.shape['x'] == (2,):
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$\\hat{y}(x)$')

                plt_data = ax.plot_surface(x[..., 0], x[..., 1], y_mean, cmap=plt.cm.viridis)
                if do_std:
                    plt_data_lo = ax.plot_surface(x[..., 0], x[..., 1], y_mean - y_std, cmap=plt.cm.viridis)
                    plt_data_hi = ax.plot_surface(x[..., 0], x[..., 1], y_mean + y_std, cmap=plt.cm.viridis)
                    plt_data = (plt_data, (plt_data_lo, plt_data_hi))
            else:
                raise ValueError("Predictor data 'x' must have shape () or (2,).")
        else:
            raise ValueError("Target data 'y' must have shape ().")

        return plt_data


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
#         self._model_gen = functools.partial(YcXModel.finite_model, supp_x=supp_x['x'], supp_y=supp_y['y'], rng=None)
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
