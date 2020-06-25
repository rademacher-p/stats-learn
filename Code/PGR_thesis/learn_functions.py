"""
Supervised learning functions.
"""

import numpy as np
import functools
from util.generic import vectorize_x_func, empirical_pmf
from loss_functions import loss_se, loss_01
from SL_obj import YcXModel

# TODO: add method functionality to work with SKL, TF conventions?
# TODO: COMPLETE property set/get check, rework!


class BaseLearner:
    def __init__(self):
        self._data_shape_x = None
        self._data_shape_y = None

        self.loss_fcn = None

    def fit(self, d):
        raise NotImplementedError("Method must be overwritten.")
        pass

    def predict(self, x):
        return vectorize_x_func(self._predict_single, x)

    def _predict_single(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass

    def evaluate(self, d):
        loss = np.array([self._evaluate_single(d_i) for d_i in d])
        return loss.mean()

    def _evaluate_single(self, d):
        return self.loss_fcn(self._predict_single(d['x']), d['y'])


class BayesLearner(BaseLearner):
    def __init__(self, bayes_model):
        super().__init__()
        self.bayes_model = bayes_model

        self._posterior_mean = None
        self.fit()

    @property
    def posterior_mean(self):
        return self._posterior_mean

    def fit(self, d=None):
        if d is None:
            d = np.array([], dtype=[('y', '<f8', self.bayes_model._data_shape_y),
                                    ('x', '<f8', self.bayes_model._data_shape_x)])
        self._posterior_mean = self.bayes_model.posterior_mean(d)


class BayesClassifier(BayesLearner):
    def __init__(self, bayes_model):
        super().__init__(bayes_model)
        self.loss_fcn = loss_01

    def _predict_single(self, x):
        return self._posterior_mean.mode_y_x(x)


class BayesEstimator(BayesLearner):
    def __init__(self, bayes_model):
        super().__init__(bayes_model)
        self.loss_fcn = loss_se

    def _predict_single(self, x):
        return self._posterior_mean.mean_y_x(x)



# class BayesLearner(BaseLearner):
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
#     def posterior_mean(self):
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
# class BayesClassifier(BayesLearner):
#     def __init__(self, supp_x, supp_y, alpha_0, mean):
#         super().__init__(supp_x, supp_y, alpha_0, mean)
#         self.loss_fcn = loss_01
#
#     def _predict_single(self, x):
#         return self._posterior_mean.mode_y_x(x)
#
#
# class BayesEstimator(BayesLearner):
#     def __init__(self, supp_x, supp_y, alpha_0, mean):
#         super().__init__(supp_x, supp_y, alpha_0, mean)
#         self.loss_fcn = loss_se
#
#     def _predict_single(self, x):
#         return self._posterior_mean.mean_y_x(x)

