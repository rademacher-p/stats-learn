from functools import cache, partial

import numpy as np
import sklearn as skl
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor

from stats_learn import spaces
from stats_learn.predictors.base import Base
from stats_learn.random.elements import Normal
from stats_learn.util import vectorize_func


class SKLRegressor(Base):
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


class GpSKLRegressor(SKLRegressor):
    def __init__(
        self,
        space,
        proc_funcs=(),
        name=None,
        gp_mean=None,
        noise_var=0.0,
        skl_kwargs=None,
    ):
        if skl_kwargs is None:
            skl_kwargs = {}
        skl_kwargs["normalize_y"] = False
        skl_kwargs["alpha"] = noise_var
        estimator = GaussianProcessRegressor(**skl_kwargs)
        super().__init__(estimator, space, proc_funcs, name)
        self.gp_mean = gp_mean  # TODO: vectorize?
        self.noise_var = noise_var

    def _fit(self, d):
        x, y = d["x"], d["y"]
        y_res = y - self.gp_mean(x)
        self.estimator.fit(x.reshape(-1, 1), y_res)

    def _predict(self, x):
        try:
            mean = self.estimator.predict(x.reshape(-1, 1))
            return mean + self.gp_mean(x)
        except NotFittedError:
            return self.gp_mean(x)


class GpSKLPredictor(GpSKLRegressor):
    def __init__(
        self,
        loss_func,
        space,
        space_pred=None,
        proc_funcs=(),
        name=None,
        gp_mean=None,
        noise_var=0.0,
        skl_kwargs=None,
    ):
        super().__init__(space, proc_funcs, name, gp_mean, noise_var, skl_kwargs)

        self.loss_func = loss_func

        if space_pred is None:
            space_pred = space["y"]
        self.space_pred = space_pred

        self._make_predict_single()

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self._make_predict_single()

    def _fit(self, d):
        super()._fit(d)
        self._make_predict_single()

    def reset(self):
        super().reset()
        self._make_predict_single()

    def _predict(self, x):
        vec_func = vectorize_func(self._predict_single, self.shape["x"])
        return vec_func(x)

    def _make_predict_single(self):
        def _fn(x):
            # try:
            #     mean, cov = self.estimator.predict(x.reshape(-1, 1), return_cov=True)
            #     mean += self.gp_mean(x)
            #     cov += self.noise_var
            # except NotFittedError:
            #     mean = self.gp_mean(x)
            #     cov = np.zeros(2 * mean.shape)
            mean, cov = self.estimator.predict(x.reshape(-1, 1), return_cov=True)
            mean += self.gp_mean(x)
            cov += self.noise_var

            model_y = Normal(mean.item(), cov.item())

            def _risk(h):
                _fn = partial(self.loss_func, h)
                return model_y.expectation(_fn)

            return self.space_pred.argmin(_risk)

        if isinstance(self.space["x"], spaces.Discrete):
            _fn = cache(_fn)

        self._predict_single = _fn
