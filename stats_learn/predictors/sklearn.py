import numpy as np
import sklearn as skl
from sklearn.exceptions import NotFittedError

from stats_learn.loss_funcs import loss_se
from stats_learn.predictors.base import Base


class SKLPredictor(Base):  # TODO: rework for new reset/fit functionality

    # FIXME: inheritance feels broken

    def __init__(self, estimator, space, proc_funcs=(), name=None):
        if skl.base.is_regressor(estimator):
            loss_func = loss_se
        else:
            raise ValueError("Estimator must be regressor-type.")

        super().__init__(loss_func, space, proc_funcs, name)

        self.estimator = estimator
        self.can_warm_start = hasattr(self.estimator, 'warm_start')  # TODO: bugged if estimator is `Pipeline`?

    @property
    def _model_obj(self):
        raise NotImplementedError

    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self.estimator, key, val)

    def reset(self):
        self.estimator = skl.base.clone(self.estimator)  # manually reset learner if `fit` is not called

    def _fit(self, d):
        x, y = d['x'].reshape(-1, 1), d['y']
        self.estimator.fit(x, y)

    # def _fit(self, d, warm_start):
    #     if hasattr(self.estimator, 'warm_start'):  # TODO: check unneeded if not warm_start
    #         self.estimator.set_params(warm_start=warm_start)
    #     elif isinstance(self.estimator, Pipeline):
    #         self.estimator.set_params(regressor__warm_start=warm_start)  # assumes pipeline step called "regressor"
    #     else:
    #         raise NotImplementedError
    #
    #     if len(d) > 0:
    #         x, y = d['x'].reshape(-1, 1), d['y']
    #         self.estimator.fit(x, y)
    #     elif not warm_start:
    #         self.estimator = skl.base.clone(self.estimator)  # manually reset learner if `fit` is not called

    def _predict(self, x):
        try:
            x = x.reshape(-1, 1)
            return self.estimator.predict(x)
        except NotFittedError:
            return np.full(x.shape[0], np.nan)