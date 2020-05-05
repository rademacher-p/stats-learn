"""
Supervised Learning base classes.
"""

# TODO: add conditional RE object?
# TODO: docstrings?

import numpy as np
from scipy.stats._multivariate import multi_rv_generic


class GenericModel(multi_rv_generic):
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

    rvs = GenericRV.rvs

    def _rvs(self, size=(), random_state=None):
        return None


class GenericModelRVx(GenericModel):
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


class GenericModelRVy(GenericModel):
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


class YcXModel(GenericModel):

    def __new__(cls, model_x, model_y_x, seed=None):
        is_numeric_y_x = isinstance(model_y_x(model_x.rvs()), GenericRV)
        if isinstance(model_x, GenericRV):
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
        self._update_attr(model_x, model_y_x)

    @property
    def model_x(self):
        return self._model_x

    @model_x.setter
    def model_x(self, model_x):
        self._update_attr(model_x=model_x)

    @property
    def model_y_x(self):
        return self._model_y_x

    @model_y_x.setter
    def model_y_x(self, model_y_x):
        self._update_attr(model_y_x=model_y_x)

    def _update_attr(self, *args, **kwargs):
        if 'model_x' in kwargs.keys():
            self._model_x = kwargs['model_x']
            self._update_x()
        elif len(args) > 0:
            self._model_x = args[0]
            self._update_x()

        if 'model_y_x' in kwargs.keys():
            self._model_y_x = kwargs['model_y_x']
            self._update_y_x()
        elif len(args) > 1:
            self._model_y_x = args[1]
            self._update_y_x()

    def _update_x(self):
        self._data_shape_x = self._model_x.data_shape
        self._mode_x = self._model_x.mode

    def _update_y_x(self):
        self._data_shape_y = self._model_y_x(self._model_x.rvs()).data_shape
        self._mode_y_x = _vectorize_func(lambda x: self._model_y_x(x).mode, self._data_shape_x)

    def _rvs(self, size=(), random_state=None):
        X = np.asarray(self.model_x.rvs(size, random_state))
        if len(size) == 0:
            Y = self.model_y_x(X).rvs(size, random_state)
            D = np.array((Y, X), dtype=[('y', Y.dtype, self.data_shape_y), ('x', X.dtype, self.data_shape_x)])
        else:
            Y = np.asarray([self.model_y_x(x).rvs((), random_state)
                            for x in X.reshape((-1,) + self._data_shape_x)]).reshape(size + self.data_shape_y)
            D = np.array(list(zip(Y.reshape((-1,) + self.data_shape_y), X.reshape((-1,) + self.data_shape_x))),
                         dtype=[('y', Y.dtype, self.data_shape_y), ('x', X.dtype, self.data_shape_x)]).reshape(size)

        return D


class YcXModelRVx(YcXModel, GenericModelRVx):
    def _update_x(self):
        super()._update_x()
        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov


class YcXModelRVy(YcXModel, GenericModelRVy):
    def _update_y_x(self):
        super()._update_y_x()
        self._mean_y_x = _vectorize_func(lambda x: self._model_y_x(x).mean, self._data_shape_x)
        self._cov_y_x = _vectorize_func(lambda x: self._model_y_x(x).cov, self._data_shape_x)


class YcXModelRVyx(YcXModelRVx, YcXModelRVy):
    pass


# theta_m = DirichletRV(8, [[.2, .1], [.3,.4]])
# def theta_c(x): return FiniteRE([[0, 1], [2, 3]], x)

# theta_m = DirichletRV(8, [[.2, .1, .1], [.3, .1, .2]])
# def theta_c(x): return FiniteRE(np.stack(np.meshgrid([0,1,2],[0,1]), axis=-1), x)

# theta_m = DirichletRV(6, [.5, .5])
# def theta_c(x): return FiniteRE(['a', 'b'], x)
# # def theta_c(x): return FiniteRE([0, 1], x)

# t = YcXModel(theta_m, theta_c)
# t.rvs()
# t.rvs(4)
# t.mode_y_x(t.model_x.rvs(4))
# t.mean_y_x(t.model_x.rvs(4))
# t.cov_y_x(t.model_x.rvs(4))