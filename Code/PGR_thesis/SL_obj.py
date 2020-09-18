"""
Supervised Learning base classes.
"""

# TODO: add conditional RE object?

import numpy as np
from scipy import stats
from scipy.stats._multivariate import multi_rv_generic

from RE_obj import NormalRV
from RE_obj_callable import BaseRE, BaseRV, FiniteRE, DirichletRV, BetaRV       # TODO: note - CALLABLE!!!!
from util.generic import vectorize_func, check_data_shape


class BaseModel(multi_rv_generic):
    """
    Base class for supervised learning data models.
    """

    def __init__(self, rng=None):
        super().__init__(rng)

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

    # @property
    # def mode_y_x(self):     # TODO: vectorization in setters?
    #     return self._mode_y_x

    def mode_y_x(self, x):        # TODO: avoid callable properties approach?
        return vectorize_func(self._mode_y_x_single, self._data_shape_x)(x)

    def _mode_y_x_single(self, x):
        raise NotImplementedError
        pass

    rvs = BaseRE.rvs

    def _rvs(self, size=(), rng=None):
        raise NotImplementedError("Method must be overwritten.")
        pass


class BaseModelRVx(BaseModel):
    def __init__(self, rng=None):
        super().__init__(rng)
        self._mean_x = None
        self._cov_x = None

    @property
    def mean_x(self):
        return self._mean_x

    @property
    def cov_x(self):
        return self._cov_x


class BaseModelRVy(BaseModel):
    def __init__(self, rng=None):
        super().__init__(rng)
        self._mean_y_x = None
        self._cov_y_x = None

    # @property
    # def mean_y_x(self):
    #     return self._mean_y_x

    def mean_y_x(self, x):
        return vectorize_func(self._mean_y_x_single, self._data_shape_x)(x)

    def _mean_y_x_single(self, x):
        raise NotImplementedError
        pass

    # @property
    # def cov_y_x(self):
    #     return self._cov_y_x

    def cov_y_x(self, x):
        return vectorize_func(self._cov_y_x_single, self._data_shape_x)(x)

    def _cov_y_x_single(self, x):
        raise NotImplementedError
        pass


class YcXModel(BaseModel):

    def __new__(cls, model_x, model_y_x, rng=None):
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

    def __init__(self, model_x, model_y_x, rng=None):
        super().__init__(rng)
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
        # self._mode_y_x = vectorize_func(lambda x: self._model_y_x(x).mode, self._data_shape_x)
        self._mode_y_x_single = lambda x: self._model_y_x(x).mode

    def _rvs(self, size=(), rng=None):
        d_x = np.array(self.model_x.rvs(size, rng))
        d_y = np.array([self.model_y_x(x).rvs((), rng)
                        for x in d_x.reshape((-1,) + self._data_shape_x)]).reshape(size + self.data_shape_y)

        # d = np.array(list(zip(d_y.reshape((-1,) + self.data_shape_y), d_x.reshape((-1,) + self.data_shape_x))),
        #              dtype=[('y', d_y.dtype, self.data_shape_y), ('x', d_x.dtype, self.data_shape_x)]).reshape(size)
        d = np.array(list(zip(d_x.reshape((-1,) + self.data_shape_x), d_y.reshape((-1,) + self.data_shape_y))),
                     dtype=[('x', d_x.dtype, self.data_shape_x), ('y', d_y.dtype, self.data_shape_y)]).reshape(size)

        return d

    @classmethod
    def finite_model_orig(cls, supp_x, p_x, supp_y, p_y_x, rng=None):        # TODO: DELETE
        model_x = FiniteRE.gen_func(supp_x, p_x)

        def model_y_x(x): return FiniteRE.gen_func(supp_y, p_y_x(x))

        return cls(model_x, model_y_x, rng)

    @classmethod
    def finite_model(cls, p_x, p_y_x, rng=None):
        model_x = FiniteRE(p_x)

        def model_y_x(x): return FiniteRE(p_y_x(x))

        return cls(model_x, model_y_x, rng)

    @classmethod
    def beta_model(cls, a, b, c, rng=None):
        model_x = BetaRV(a, b)

        def model_y_x(x): return BetaRV(c*x, c*(1-x))

        return cls(model_x, model_y_x, rng)

    # TODO: subclass, overwrite methods for efficiency?

    @classmethod
    def norm_model(cls, model_x=NormalRV(), basis_y_x=(lambda x: 1,), weights=(0,), cov_y_x=1, rng=None):

        def model_y_x(x):
            mean_y_x = sum(weight * func(x) for weight, func in zip(weights, basis_y_x))
            return NormalRV(mean_y_x, cov_y_x)

        return cls(model_x, model_y_x, rng)


class YcXModelRVx(YcXModel, BaseModelRVx):
    def _update_x(self):
        super()._update_x()
        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov


class YcXModelRVy(YcXModel, BaseModelRVy):
    def _update_y_x(self):
        super()._update_y_x()
        # self._mean_y_x = vectorize_func(lambda x: self._model_y_x(x).mean, self._data_shape_x)
        # self._cov_y_x = vectorize_func(lambda x: self._model_y_x(x).cov, self._data_shape_x)
        self._mean_y_x_single = lambda x: self._model_y_x(x).mean
        self._cov_y_x_single = lambda x: self._model_y_x(x).cov


class YcXModelRVyx(YcXModelRVx, YcXModelRVy):
    pass


# theta_m = DirichletRV(8, [[.2, .1], [.3, .4]])
# def theta_c(x): return FiniteRE.gen_func([[0, 1], [2, 3]], x)
#
# theta_m = DirichletRV(8, [[.2, .1, .1], [.3, .1, .2]])
# def theta_c(x): return FiniteRE.gen_func(np.stack(np.meshgrid([0,1,2],[0,1]), axis=-1), x)
#
# theta_m = DirichletRV(6, [.5, .5])
# # def theta_c(x): return FiniteRE.gen_func(['a', 'b'], x)
# def theta_c(x): return FiniteRE.gen_func([0, 1], x)
#
# t = YcXModel(theta_m, theta_c)
# t.rvs()
# t.rvs(4)
# t.mode_y_x(t.model_x.rvs(4))
# t.mean_y_x(t.model_x.rvs(4))
# t.cov_y_x(t.model_x.rvs(4))


class NormalRVModel(BaseModelRVx, BaseModelRVy):
    def __init__(self, model_x=NormalRV(), basis_y_x=(lambda x: 1.,), weights=(0.,),
                 cov_y_x=1., rng=None):
        super().__init__(rng)

        self.model_x = model_x
        self.basis_y_x = basis_y_x
        self.weights = weights

        # self._cov_y_x_const = np.array(cov_y_x)
        self._cov_y_x_single = lambda x: np.array(cov_y_x)

        # _temp = self._cov_y_x_const.shape
        _temp = self._cov_y_x_single(model_x.rvs()).shape
        self._data_shape_y = _temp[:int(len(_temp) / 2)]

        self._mode_y_x_single = self._mean_y_x_single

    @property
    def model_x(self):
        return self._model_x

    @model_x.setter
    def model_x(self, model_x):
        self._model_x = model_x

        self._data_shape_x = self._model_x.data_shape
        self._mode_x = self._model_x.mode

        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov

    def model_y_x(self, x):
        mean = self._mean_y_x_single(x)
        # cov = self._cov_y_x_const
        cov = self._cov_y_x_single(x)
        return NormalRV(mean, cov)

    # @BaseModelRVy.cov_y_x.setter
    # def cov_y_x(self, cov_y_x):
    #     self._cov_y_x = np.array(cov_y_x)
    #     _temp = self._cov_y_x.shape
    #     self._data_shape_y = _temp[:int(len(_temp) / 2)]

    def _mean_y_x_single(self, x):
        return sum(weight * func(x) for weight, func in zip(self.weights, self.basis_y_x))

    # def _cov_y_x_single(self, x):
    #     return self._cov_y_x_const

    _rvs = YcXModel._rvs


# g = NormalRVModel(basis_y_x=(lambda x: x,), weights=(1,), cov_y_x=.01)
# r = g.rvs(100)
# # plt.plot(r['x'], r['y'], '.')
