"""
Supervised learning functions.
"""

import numpy as np
import functools
from util.util import vectorize_x_func, empirical_pmf
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


class DirichletClassifier(BaseLearner):
    def __init__(self, supp_x, supp_y, alpha_0, mean):
        super().__init__()
        self.loss_fcn = loss_se

        self.supp_x = supp_x        # TODO: Assumed to be my SL structured arrays!
        self.supp_y = supp_y

        self._supp_shape_x = supp_x.shape
        self._supp_shape_y = supp_y.shape
        self._data_shape_x = supp_x.dtype['x'].shape
        self._data_shape_y = supp_y.dtype['y'].shape

        self.alpha_0 = alpha_0
        self.mean = mean

        self._mean_x = mean.reshape(self._supp_shape_x + (-1,)).sum(axis=-1)

        def _mean_y_x(x):
            _mean_flat = mean.reshape((-1,) + self._supp_shape_y)
            _mean_slice = _mean_flat[np.all(x.flatten()
                                     == self.supp_x['x'].reshape(self.supp_x.size, -1), axis=-1)].squeeze(axis=0)
            mean_y = _mean_slice / _mean_slice.sum()
            return mean_y

        self._mean_y_x = _mean_y_x

        self._posterior = None
        self._model_gen = functools.partial(YcXModel.finite_model, supp_x=supp_x['x'], supp_y=supp_y['y'])

    @property
    def mean_x(self):
        return self._mean_x

    @property
    def posterior(self):
        return self._posterior

    def fit(self, d):
        emp_dist_x = empirical_pmf(d['x'], self.supp_x, self._data_shape_x)

        def emp_dist_y_x(x):
            d_match = d[np.all(x.flatten() == d['x'].reshape(len(d), -1), axis=-1)].squeeze()
            return empirical_pmf(d_match, self.supp_y, self._data_shape_y)

        # TODO: INCOMPLETE!!!!

        p_x = None
        p_y_x = None
        self._posterior = self._model_gen(p_x=p_x, p_y_x=p_y_x)

