import numpy as np
import sklearn as skl
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor

from stats_learn.predictors.base import Base


class SKLPredictor(Base):
    # TODO: rework for new reset/fit functionality
    # FIXME: inheritance feels broken

    def __init__(self, estimator, space, proc_funcs=(), name=None):
        super().__init__(space, proc_funcs, name)

        self.estimator = estimator
        # self.can_warm_start = hasattr(self.estimator, "warm_start")
        # FIXME: bugged if estimator is `Pipeline`?

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.estimator, key, value)

    def reset(self):
        # manually reset learner if `fit` is not called
        self.estimator = skl.base.clone(self.estimator)

    def _fit(self, d):
        x, y = d["x"].reshape(-1, 1), d["y"]
        self.estimator.fit(x, y)

    # def _fit(self, d, warm_start):
    #     if hasattr(self.estimator, "warm_start"):
    #         # TODO: check unneeded if not warm_start
    #         self.estimator.set_params(warm_start=warm_start)
    #     elif isinstance(self.estimator, Pipeline):
    #         self.estimator.set_params(regressor__warm_start=warm_start)
    #         # assumes pipeline step called "regressor"
    #     else:
    #         raise NotImplementedError

    #     if len(d) > 0:
    #         x, y = d["x"].reshape(-1, 1), d["y"]
    #         self.estimator.fit(x, y)
    #     elif not warm_start:
    #         self.estimator = skl.base.clone(self.estimator)
    #         # manually reset learner if `fit` is not called

    def _predict(self, x):
        try:
            x = x.reshape(-1, 1)
            return self.estimator.predict(x)
        except NotFittedError:
            return np.full(x.shape[0], np.nan)


class GpSKLPredictor(SKLPredictor):
    def __init__(self, space, proc_funcs=(), name=None, skl_kwargs=None, gp_mean=None):
        if skl_kwargs is None:
            skl_kwargs = {}
        skl_kwargs["normalize_y"] = False
        estimator = GaussianProcessRegressor(**skl_kwargs)
        super().__init__(estimator, space, proc_funcs, name)
        self.gp_mean = gp_mean  # TODO: vectorize?

    def _fit(self, d):
        x, y = d["x"], d["y"]
        y_res = y - self.gp_mean(x)
        self.estimator.fit(x.reshape(-1, 1), y_res)

    def _predict(self, x):
        try:
            y_res = self.estimator.predict(x.reshape(-1, 1))
            y = y_res + self.gp_mean(x)
            return y
        except NotFittedError:
            return self.gp_mean(x)
