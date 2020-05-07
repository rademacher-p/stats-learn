"""
Supervised Learning base classes.
"""

# TODO: add conditional RE object?
# TODO: docstrings?

import numpy as np
from scipy.stats._multivariate import multi_rv_generic

from RE_obj import BaseRV
from util.util import vectorize_x_func


class BaseModel(multi_rv_generic):
    """
    Base class for supervised learning data models.
    """

    def __init__(self, seed=None):
        super().__init__(seed)

        self._data_shape_x = None
        self._data_shape_y = None

        self._mode_x = None
        self._mode_y_x = None

    @property
    def data_shape_x(self):
        return self._data_shape_x

    @property
    def data_shape_y(self):
        return self._data_shape_y

    @property
    def mode_x(self):
        return self._mode_x

    @property
    def mode_y_x(self):
        return self._mode_y_x

    rvs = BaseRV.rvs

    def _rvs(self, size=(), random_state=None):
        raise NotImplementedError("Method must be overwritten.")
        pass


class BaseModelRVx(BaseModel):
    """
    Base
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self._mean_x = None
        self._cov_x = None

    @property
    def mean_x(self):
        return self._mean_x

    @property
    def cov_x(self):
        return self._cov_x


class BaseModelRVy(BaseModel):
    """
    Base
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self._mean_y_x = None
        self._cov_y_x = None

    @property
    def mean_y_x(self):
        return self._mean_y_x

    @property
    def cov_y_x(self):
        return self._cov_y_x


class YcXModel(BaseModel):

    def __new__(cls, model_x, model_y_x, seed=None):
        is_numeric_y_x = isinstance(model_y_x(model_x.rvs()), BaseRV)
        if isinstance(model_x, BaseRV):
            if is_numeric_y_x:
                return super().__new__(YcXModelRVyx)
            else:
                return super().__new__(YcXModelRVx)
        else:
            if is_numeric_y_x:
                return super().__new__(YcXModelRVy)
            else:
                return super().__new__(cls)

    def __init__(self, model_x, model_y_x, seed=None):
        super().__init__(seed)
        self._model_x = model_x
        self._update_x()
        self._model_y_x = model_y_x
        self._update_y_x()

    @property
    def model_x(self):
        return self._model_x

    @model_x.setter
    def model_x(self, model_x):
        self._model_x = model_x
        self._update_x()

    @property
    def model_y_x(self):
        return self._model_y_x

    @model_y_x.setter
    def model_y_x(self, model_y_x):
        self._model_y_x = model_y_x
        self._update_y_x()

    def _update_x(self):
        self._data_shape_x = self._model_x.data_shape
        self._mode_x = self._model_x.mode

    def _update_y_x(self):
        self._data_shape_y = self._model_y_x(self._model_x.rvs()).data_shape
        self._mode_y_x = vectorize_x_func(lambda x: self._model_y_x(x).mode, self._data_shape_x)

    def _rvs(self, size=(), random_state=None):
        d_x = np.asarray(self.model_x.rvs(size, random_state))
        if len(size) == 0:
            d_y = self.model_y_x(d_x).rvs(size, random_state)
            d = np.array((d_y, d_x), dtype=[('y', d_y.dtype, self.data_shape_y), ('x', d_x.dtype, self.data_shape_x)])
        else:
            d_y = np.asarray([self.model_y_x(x).rvs((), random_state)
                              for x in d_x.reshape((-1,) + self._data_shape_x)]).reshape(size + self.data_shape_y)
            d = np.array(list(zip(d_y.reshape((-1,) + self.data_shape_y), d_x.reshape((-1,) + self.data_shape_x))),
                         dtype=[('y', d_y.dtype, self.data_shape_y), ('x', d_x.dtype, self.data_shape_x)]).reshape(size)

        return d

    @classmethod
    def finite_model(cls):
        return None     # TODO: COMPLETE!


class YcXModelRVx(YcXModel, BaseModelRVx):
    def _update_x(self):
        super()._update_x()
        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov


class YcXModelRVy(YcXModel, BaseModelRVy):
    def _update_y_x(self):
        super()._update_y_x()
        self._mean_y_x = vectorize_x_func(lambda x: self._model_y_x(x).mean, self._data_shape_x)
        self._cov_y_x = vectorize_x_func(lambda x: self._model_y_x(x).cov, self._data_shape_x)


class YcXModelRVyx(YcXModelRVx, YcXModelRVy):
    pass


# theta_m = DirichletRV(8, [[.2, .1], [.3,.4]])
# def theta_c(x): return FiniteRE([[0, 1], [2, 3]], x)
#
# theta_m = DirichletRV(8, [[.2, .1, .1], [.3, .1, .2]])
# def theta_c(x): return FiniteRE(np.stack(np.meshgrid([0,1,2],[0,1]), axis=-1), x)
#
# theta_m = DirichletRV(6, [.5, .5])
# def theta_c(x): return FiniteRE(['a', 'b'], x)
# # def theta_c(x): return FiniteRE([0, 1], x)
#
# t = YcXModel(theta_m, theta_c)
# t.rvs()
# t.rvs(4)
# t.mode_y_x(t.model_x.rvs(4))
# t.mean_y_x(t.model_x.rvs(4))
# t.cov_y_x(t.model_x.rvs(4))
