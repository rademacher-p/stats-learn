"""
Supervised learning functions.
"""

import functools
from math import floor

import numpy as np
import matplotlib.pyplot as plt

from util.generic import vectorize_func, empirical_pmf
from loss_funcs import loss_se, loss_01
from SL_obj import YcXModel

# TODO: add method functionality to work with SKL, TF conventions?
# TODO: COMPLETE property set/get check, rework!


# FIXME FIXME: infer learner type from loss_func? use same model attribute for both optimal and bayesian learners!


#%% Decision Functions

class BaseDecisionFunc:
    # def __new__(cls, loss_func):
    #     if loss_func == loss_01:
    #         return super().__new__(BaseClassifier)
    #     elif loss_func == loss_se:
    #         return super().__new__(BaseRegressor)
    #     else:
    #         return super().__new__(cls)

    def __init__(self):
        self.model = None
        self.loss_func = None

        self._data_shape_x = None
        self._data_shape_y = None

    @property
    def data_shape_x(self):
        return self._data_shape_x

    @property
    def data_shape_y(self):
        return self._data_shape_y

    def predict(self, x):
        return vectorize_func(self._predict_single, data_shape=self._data_shape_x)(x)

    def _predict_single(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass

    def evaluate(self, d):
        # loss = np.array([self._evaluate_single(d_i['x'], d_i['y']) for d_i in d])
        # loss = np.array([self._evaluate_single(x, y) for x, y in zip(d['x'], d['y'])])
        loss = np.array([self.loss_func(h, y) for h, y in zip(self.predict(d['x']), d['y'])])
        return loss.mean()

    # def _evaluate_single(self, x, y):
    #     return self.loss_func(self._predict_single(x), y)


class BaseClassifier(BaseDecisionFunc):
    def __init__(self):
        super().__init__()
        self.loss_func = loss_01

    def predict(self, x):
        return self.model.mode_y_x(x)


class BaseRegressor(BaseDecisionFunc):
    def __init__(self):
        super().__init__()
        self.loss_func = loss_se

    def predict(self, x):
        return self.model.mean_y_x(x)


class DecisionFromModel(BaseDecisionFunc):
    def __init__(self, model, loss_func):
        super().__init__(loss_func)
        self.model = model

        self._data_shape_x = self.model.data_shape_x
        self._data_shape_y = self.model.data_shape_y


#%% Learning Functions

class BaseLearner(BaseDecisionFunc):
    def fit(self, d):
        raise NotImplementedError("Method must be overwritten.")
        pass


class BaseBayesLearner(BaseLearner):
    def __init__(self, bayes_model):
        super().__init__()
        self.bayes_model = bayes_model
        self.model = None

        self._data_shape_x = self.bayes_model.data_shape_x
        self._data_shape_y = self.bayes_model.data_shape_y

        self.prior = self.bayes_model.prior
        self.posterior = None
        self.predictive_dist = None

        self.fit()

    def fit(self, d=None):
        if d is None:
            d = np.array([], dtype=[('x', '<f8', self._data_shape_x),
                                    ('y', '<f8', self._data_shape_y)])

        self.posterior, self.predictive_dist, self.model = self.bayes_model.fit(d)

    def plot_param_dist(self, ax_prior=None, ax_posterior=None):    # TODO: improve or delete
        plt_prior = self.prior.plot_pf(ax=ax_prior)
        ax_prior = plt_prior.axes
        ax_posterior = ax_prior
        self.posterior.plot_pf(ax=ax_posterior)

    def plot_prediction(self, x_plt, ax=None):
        # self.predictive_dist.plot_pf(ax=ax)

        # plt_data = self.posterior_model.model_x.plot_pf()
        # ax = plt_data.axes
        # plt_data.remove()

        if ax is None:
            _, ax = plt.subplots()
            ax.set(xlabel='$x$', ylabel='$\hat{y}$')
        plt_data = ax.plot(x_plt, self.predict(x_plt))

        return plt_data


# class BayesClassifier(BaseBayesLearner):
#     def __init__(self, bayes_model):
#         super().__init__(bayes_model)
#
#     def predict(self, x):
#         return self.model.mode_y_x(x)
#
#     # def _predict_single(self, x):
#     #     # return self.posterior_model._mode_y_x_single(x)
#     #     return self.predictive_dist(x).mode    # TODO: argmax?
#
#
# class BayesEstimator(BaseBayesLearner):
#     def __init__(self, bayes_model):
#         super().__init__(bayes_model)
#
#     def predict(self, x):
#         return self.model.mean_y_x(x)
#
#     # def _predict_single(self, x):
#     #     # return self.posterior_model._mean_y_x_single(x)
#     #     return self.predictive_dist(x).mean        # TODO: m1?


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


class BetaEstimatorTemp(BaseLearner):
    def __init__(self, n_x=10):
        super().__init__()
        self.loss_fcn = loss_se
        self.n_x = n_x
        self.avg_y_x = np.zeros(n_x)

    def fit(self, d):
        delta = 1 / self.n_x
        for i in range(self.n_x):
            flag_match = np.logical_and(d['x'] >= i * delta, d['x'] < (i + 1) * delta)
            if flag_match.any():
                self.avg_y_x[i] = d[flag_match]['y'].mean()

    def _predict_single(self, x):
        i = floor(x * self.n_x)
        return self.avg_y_x[i]


# class BaseBayesLearner(BaseLearner):
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
# class BayesClassifier(BaseBayesLearner):
#     def __init__(self, supp_x, supp_y, alpha_0, mean):
#         super().__init__(supp_x, supp_y, alpha_0, mean)
#         self.loss_func = loss_01
#
#     def _predict_single(self, x):
#         return self._posterior_mean.mode_y_x(x)
#
#
# class BayesEstimator(BaseBayesLearner):
#     def __init__(self, supp_x, supp_y, alpha_0, mean):
#         super().__init__(supp_x, supp_y, alpha_0, mean)
#         self.loss_func = loss_se
#
#     def _predict_single(self, x):
#         return self._posterior_mean.mean_y_x(x)

