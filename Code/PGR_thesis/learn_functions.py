"""
Supervised learning functions.
"""

import numpy as np
from util.util import vectorize_x_func, empirical_pmf

# TODO: add method functionality to work with SKL, TF conventions?


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
    def __init__(self, dirichlet_rv):
        super().__init__()

        # self.model = dirichlet_rv     # TODO: complete