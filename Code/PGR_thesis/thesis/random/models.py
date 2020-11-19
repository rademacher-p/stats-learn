"""
SL models.
"""

import math
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from thesis.random import elements as rand_elements
# from thesis._deprecated.RE_obj_callable import FiniteRE  # TODO: note - CALLABLE!!!!

from thesis.util.generic import RandomGeneratorMixin, vectorize_func, vectorize_func_dec
from thesis.util import spaces

# TODO: add marginal/conditional pf methods


class Base(RandomGeneratorMixin):
    """
    Base class for supervised learning data models.
    """

    def __init__(self, rng=None):
        super().__init__(rng)
        self._space = {'x': None, 'y': None}

        # self.model_x = None
        # self.model_y_x = None

        self._mode_x = None

    space = property(lambda self: self._space)

    shape = property(lambda self: {key: space.shape for key, space in self._space.items()})
    size = property(lambda self: {key: space.size for key, space in self._space.items()})
    ndim = property(lambda self: {key: space.ndim for key, space in self._space.items()})

    dtype = property(lambda self: {key: space.dtype for key, space in self._space.items()})

    # shape = property(lambda self: self._shape)
    # size = property(lambda self: {key: math.prod(val) for key, val in self._shape.items()})
    # ndim = property(lambda self: {key: len(val) for key, val in self._shape.items()})

    # TODO: default stats to reference `model_x` and `model_y_x` attributes

    # @property
    # def model_x(self):
    #     return self._model_x

    @property
    def mode_x(self):
        return self._mode_x

    def mode_y_x(self, x):
        return vectorize_func(self._mode_y_x_single, self.shape['x'])(x)

    def _mode_y_x_single(self, x):
        raise NotImplementedError
        pass

    rvs = rand_elements.Base.rvs

    def _rvs(self, n, rng):
        d_x = np.array(self.model_x.rvs(n, rng=rng))
        d_y = np.stack([self.model_y_x(x).rvs(rng=rng) for x in d_x])

        d = np.array(list(zip(d_x, d_y)), dtype=[('x', d_x.dtype, self.shape['x']), ('y', d_y.dtype, self.shape['y'])])

        return d


class MixinRVx:
    _mean_x: Optional[np.ndarray]
    _cov_x: Optional[np.ndarray]

    @property
    def mean_x(self):
        return self._mean_x

    @property
    def cov_x(self):
        return self._cov_x


class MixinRVy:
    shape: dict

    def mean_y_x(self, x):
        return vectorize_func(self._mean_y_x_single, self.shape['x'])(x)

    def _mean_y_x_single(self, x):
        raise NotImplementedError
        pass

    def cov_y_x(self, x):
        return vectorize_func(self._cov_y_x_single, self.shape['x'])(x)

    def _cov_y_x_single(self, x):
        raise NotImplementedError
        pass


class DataConditional(Base):
    def __new__(cls, model_x, model_y_x, rng=None):
        is_numeric_y = isinstance(model_y_x(model_x.rvs()), rand_elements.MixinRV)
        if isinstance(model_x, rand_elements.MixinRV):
            if is_numeric_y:
                return super().__new__(DataConditionalRVxy)
            else:
                return super().__new__(DataConditionalRVx)
        else:
            if is_numeric_y:
                return super().__new__(DataConditionalRVy)
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
        self._space['x'] = self._model_x.space
        self._mode_x = self._model_x.mode

    def _update_y_x(self):
        self._space['y'] = self._model_y_x(self._model_x.rvs()).space
        # self._mode_y_x = vectorize_func(lambda x: self._model_y_x(x).mode, self.shape['x'])
        # self._mode_y_x_single = lambda x: self._model_y_x(x).mode

    def _mode_y_x_single(self, x):
        return self._model_y_x(x).mode

    @classmethod
    def finite_model(cls, supp_x, p_x, supp_y, p_y_x, rng=None):
        model_x = rand_elements.Finite(supp_x, p_x)

        def model_y_x(x): return rand_elements.Finite(supp_y, p_y_x(x))

        return cls(model_x, model_y_x, rng)

    # @classmethod
    # def finite_model_orig(cls, supp_x, p_x, supp_y, p_y_x, rng=None):
    #     model_x = FiniteRE.gen_func(supp_x, p_x)
    #
    #     def model_y_x(x): return FiniteRE.gen_func(supp_y, p_y_x(x))
    #
    #     return cls(model_x, model_y_x, rng)
    #
    # @classmethod
    # def finite_model(cls, p_x, p_y_x, rng=None):
    #     model_x = FiniteRE(p_x)
    #
    #     def model_y_x(x): return FiniteRE(p_y_x(x))
    #
    #     return cls(model_x, model_y_x, rng)

    # @classmethod
    # def beta_model(cls, a, b, c, rng=None):
    #     model_x = BetaRV(a, b)
    #
    #     def model_y_x(x): return BetaRV(c*x, c*(1-x))
    #
    #     return cls(model_x, model_y_x, rng)


class DataConditionalRVx(MixinRVx, DataConditional):
    def _update_x(self):
        super()._update_x()
        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov


class DataConditionalRVy(MixinRVy, DataConditional):
    def _update_y_x(self):
        super()._update_y_x()
        # self._mean_y_x = vectorize_func(lambda x: self._model_y_x(x).mean, self.shape['x'])
        # self._cov_y_x = vectorize_func(lambda x: self._model_y_x(x).cov, self.shape['x'])
        # self._mean_y_x_single = lambda x: self._model_y_x(x).mean
        # self._cov_y_x_single = lambda x: self._model_y_x(x).cov

    def mean_y_x(self, x):
        return vectorize_func(self._mean_y_x_single, self.shape['x'])(x)

    def _mean_y_x_single(self, x):
        return self._model_y_x(x).mean

    def _cov_y_x_single(self, x):
        return self._model_y_x(x).cov


class DataConditionalRVxy(DataConditionalRVy, DataConditionalRVx):
    pass


# theta_m = Dirichlet(8, [[.2, .1], [.3, .4]])
# def theta_c(x): return Finite.gen_func([[0, 1], [2, 3]], x)
#
# theta_m = Dirichlet(8, [[.2, .1, .1], [.3, .1, .2]])
# def theta_c(x): return Finite.gen_func(np.stack(np.meshgrid([0,1,2],[0,1]), axis=-1), x)
#
# theta_m = Dirichlet(6, [.5, .5])
# # def theta_c(x): return Finite.gen_func(['a', 'b'], x)
# def theta_c(x): return Finite.gen_func([0, 1], x)
#
# t = DataConditional(theta_m, theta_c)
# t.rvs()
# t.rvs(4)
# t.mode_y_x(t.model_x.rvs(4))
# t.mean_y_x(t.model_x.rvs(4))
# t.cov_y_x(t.model_x.rvs(4))


class ClassConditional(MixinRVx, Base):
    def __init__(self, dists, model_y, rng=None):
        super().__init__(rng)

        self._dists = list(dists)
        self.model_y = model_y

        self._space['y'] = self.model_y.space
        if self.space['y'].ndim != 0 or not np.issubdtype(self.space['y'].dtype, 'U'):
            raise ValueError

        self._space['x'] = self.dists[0].space
        if not all(dist.space == self._space['x'] for dist in self.dists[1:]):
            raise ValueError("All distributions must have the same space.")

        self._update_attr()

    @classmethod
    def from_finite(cls, dists, y, p_y=None, rng=None):
        model_y = rand_elements.Finite(np.array(y, dtype='U').flatten(), p_y)
        return cls(dists, model_y, rng)

    @property
    def dists(self):
        return self._dists

    @property
    def p_y(self):
        return self.model_y.p

    @p_y.setter
    def p_y(self, val):
        self.model_y.p = val
        self._update_attr()

    def set_dist_attr(self, idx, **dist_kwargs):
        for key, val in dist_kwargs.items():
            setattr(self._dists[idx], key, val)
        self._update_attr()

    def _update_attr(self):
        self.model_x = rand_elements.MixtureRV(self.dists, self.p_y)
        self._mode_x = self.model_x.mode
        self._mean_x = self.model_x.mean
        self._cov_x = self.model_x.cov

    def _mode_y_x_single(self, x):
        return self.model_y_x(x).mode

    def model_y_x(self, x):
        temp = np.array([dist.pf(x) for dist in self.dists])
        p_y_x = temp / temp.sum()
        return rand_elements.Finite(self.model_y.supp, p_y_x)


# m = ClassConditional.from_finite([rand_elements.Normal(mean) for mean in [0, 4]], ['a', 'b'])


class NormalRegressor(MixinRVx, MixinRVy, Base):        # TODO: rename NormalLinear?
    def __init__(self, weights=(0.,), basis_y_x=None, cov_y_x=1., model_x=rand_elements.Normal(), rng=None):
        super().__init__(rng)

        self.model_x = model_x
        self.weights = weights
        self.cov_y_x_ = cov_y_x

        self._mode_y_x_single = self._mean_y_x_single

        if basis_y_x is None:
            def power_func(i):
                return lambda x: np.full(self.shape['y'], (x ** i).sum())

            self.basis_y_x = tuple(power_func(i) for i in range(len(self.weights)))
        else:
            self.basis_y_x = basis_y_x

    def __repr__(self):
        return f"NormalModel(model_x={self.model_x}, basis_y_x={self.basis_y_x}, " \
               f"weights={self.weights}, cov_y_x={self._cov_repr})"

    @property
    def model_x(self):
        return self._model_x

    @model_x.setter
    def model_x(self, model_x):
        self._model_x = model_x

        self._space['x'] = self._model_x.space
        self._mode_x = self._model_x.mode

        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, val):
        self._weights = np.array(val)
        # self._weights = np.broadcast_to(val, (self.n_weights,))

    # n_weights = property(lambda self: self._weights.size)

    @property
    def cov_y_x_(self):
        return self._cov_repr
        # return vectorize_func(self._cov_y_x_single, self.shape['x'])

    @cov_y_x_.setter
    def cov_y_x_(self, val):
        if callable(val):
            self._cov_repr = val
            self._cov_y_x_single = val
            _temp = self._cov_y_x_single(self.model_x.rvs()).shape
        else:
            self._cov_repr = np.array(val)
            self._cov_y_x_single = lambda x: self._cov_repr
            _temp = self._cov_repr.shape

        self._space['y'] = spaces.Euclidean(_temp[:int(len(_temp) / 2)])

    def _mean_y_x_single(self, x):
        return sum(weight * func(x) for weight, func in zip(self.weights, self.basis_y_x))

    def model_y_x(self, x):
        mean = self._mean_y_x_single(x)
        cov = self._cov_y_x_single(x)
        return rand_elements.Normal(mean, cov)


# g = NormalRegressor(basis_y_x=(lambda x: x,), weights=(1,), cov_y_x_single=.01, model_x=rand_elements.Normal(4))
# r = g.rvs(100)
# # plt.plot(r['x'], r['y'], '.')


class GenericEmpirical(Base):
    def __new__(cls, d, space=None, rng=None):
        dtype = np.array(d).dtype
        if np.issubdtype(dtype['x'].base, np.number):
            if np.issubdtype(dtype['y'].base, np.number):
                return super().__new__(GenericEmpiricalRVxy)
            else:
                return super().__new__(GenericEmpiricalRVx)
        else:
            if np.issubdtype(dtype['y'].base, np.number):
                return super().__new__(GenericEmpiricalRVy)
            else:
                return super().__new__(cls)

        # if all(np.issubdtype(dtype[c].base, np.number) for c in 'xy'):
        #     return super().__new__(GenericEmpiricalRVxy)
        # else:
        #     return super().__new__(cls)

    def __init__(self, d, space=None, rng=None):
        super().__init__(rng)

        d = np.array(d)
        if space is None:
            dtype = np.array(d).dtype
            for c in 'xy':
                if np.issubdtype(dtype[c].base, np.number):
                    self._space[c] = spaces.Euclidean(dtype[c].shape)
                else:
                    raise NotImplementedError
        else:
            self._space = space

        self.n = len(d)
        self.data = self._count_data(d)

        self._update_attr()

    def __repr__(self):
        return f"GenericEmpirical(space={self.space}, n={self.n})"

    def _count_data(self, d):
        values, counts = np.unique(d, return_counts=True, axis=0)
        return np.array(list(zip(values['x'], values['y'], counts)), dtype=[('x', self.dtype['x'], self.shape['x']),
                                                                            ('y', self.dtype['y'], self.shape['y']),
                                                                            ('n', np.int,)])
    # def _get_idx_x(self, x):
    #     idx = np.flatnonzero(np.all(x == self.data_x['x'], axis=tuple(range(1, 1 + self.ndim['x']))))
    #     if idx.size == 1:
    #         return idx.item()
    #     elif idx.size == 0:
    #         return None
    #     else:
    #         raise ValueError

    def _get_data_y_x(self, x):
        idx = np.flatnonzero(np.all(x == self.data_x['x'], axis=tuple(range(1, 1 + self.ndim['x']))))
        if idx.size == 1:
            return self.data_y_x[idx.item()]
        elif idx.size == 0:
            return None
        else:
            raise ValueError

    def add_data(self, d):
        self.n += len(d)

        data_new = self._count_data(d)
        idx_new = []
        for i, value in enumerate(data_new):
            idx = np.flatnonzero(value[['x', 'y']] == self.data[['x', 'y']])
            if idx.size == 1:
                self.data['n'][idx.item()] += value['n']
            else:
                idx_new.append(i)

        if len(idx_new) > 0:
            self.data = np.concatenate((self.data, data_new[idx_new]))

        self._update_attr()

    def _update_attr(self):
        self._p = self.data['n'] / self.n

        values_x, self._idx_inv = np.unique(self.data['x'], return_inverse=True, axis=0)
        self.data_y_x = []
        counts_x = np.empty(len(values_x), dtype=np.int)
        for i in range(len(values_x)):
            data_match = self.data[['y', 'n']][self._idx_inv == i]
            counts_x[i] = data_match['n'].sum()

            self.data_y_x.append(data_match)

        self.data_x = np.array(list(zip(values_x, counts_x)),
                               dtype=[('x', self.dtype['x'], self.shape['x']), ('n', np.int,)])

        self._p_x = self.data_x['n'] / self.n
        self._mode_x = self.data_x['x'][self.data_x['n'].argmax()]

    def _mode_y_x_single(self, x):
        data_ = self._get_data_y_x(x)
        if data_ is not None:
            return data_['y'][data_['n'].argmax()]
        else:
            return np.nan     # TODO: value?

    def _rvs(self, size, rng):
        return rng.choice(self.data[['x', 'y']], size, p=self._p)


class GenericEmpiricalRVx(MixinRVx, GenericEmpirical):
    def __repr__(self):
        return f"GenericEmpiricalRVx(space={self.space}, n={self.n})"

    def _update_attr(self):
        super()._update_attr()

        self._mean_x = np.tensordot(self._p_x, self.data_x['x'], axes=[0, 0])

        self._cov_x = sum(p_i * np.tensordot(ctr_i, ctr_i, 0)
                          for p_i, ctr_i in zip(self._p_x, self.data_x['x'] - self._mean_x))


class GenericEmpiricalRVy(MixinRVy, GenericEmpirical):
    def __repr__(self):
        return f"GenericEmpiricalRVy(space={self.space}, n={self.n})"

    def _mean_y_x_single(self, x):
        data_ = self._get_data_y_x(x)
        if data_ is not None:
            p_y_x = data_['n'] / data_['n'].sum()
            return np.tensordot(p_y_x, data_['y'], axes=[0, 0])
        else:
            return np.nan


class GenericEmpiricalRVxy(GenericEmpiricalRVx, GenericEmpiricalRVy):
    def __repr__(self):
        return f"GenericEmpiricalRVxy(space={self.space}, n={self.n})"


# x_ = [10, 11]
# p_x = [.5, .5]
# # y_ = ['a', 'b']
# y_ = [0, 1]
# def p_y_x_(x):
#     return [.8, .2] if x == 0 else [.2, .8]
# r = DataConditional.finite_model(x_, p_x, y_, p_y_x_)
#
# # r = NormalRegressor(weights=[1, 1], cov_y_x_single=np.eye(2))
# # r = NormalRegressor(weights=[1, 1], cov_y_x_single=1., model_x=rand_elements.Normal([0, 0]))
#
# d_ = r.rvs(10)
# e = GenericEmpirical(d_, space=r.space)
# e.add_data(r.rvs(5))
# print(e)
# pass


class Mixture(Base):
    def __new__(cls, dists, weights, rng=None):
        if all(isinstance(dist, MixinRVx) for dist in dists):
            if all(isinstance(dist, MixinRVy) for dist in dists):
                return super().__new__(MixtureRVxy)
            else:
                return super().__new__(MixtureRVx)
        else:
            if all(isinstance(dist, MixinRVy) for dist in dists):
                return super().__new__(MixtureRVy)
            else:
                return super().__new__(cls)

    def __init__(self, dists, weights, rng=None):       # TODO: special implementation for Finite? get modes, etc?
        super().__init__(rng)
        self._dists = list(dists)

        self._space = self.dists[0].space
        if not all(dist.space == self.space for dist in self.dists[1:]):
            raise ValueError("All distributions must have the same space.")

        self.weights = weights

    def __repr__(self):
        _str = "; ".join([f"{prob}: {dist}" for dist, prob in zip(self.dists, self._p)])
        return f"Mixture({_str})"

    dists = property(lambda self: self._dists)
    n_dists = property(lambda self: len(self.dists))

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = list(value)
        if len(self._weights) != self.n_dists:
            raise ValueError(f"Weights must have length {self.n_dists}.")

        self._update_attr()

    def set_dist_attr(self, idx, **dist_kwargs):      # TODO: improved implementation w/ direct self.dists access?
        for key, val in dist_kwargs.items():
            setattr(self._dists[idx], key, val)
        self._update_attr()

    def add_dist(self, dist, weight):       # TODO: type check?
        self._dists.append(dist)
        self.weights.append(weight)
        self._update_attr()

    def set_dist(self, idx, dist, weight):
        self._dists[idx] = dist
        self.weights[idx] = weight
        self._update_attr()     # weights setter not invoked

    def del_dist(self, idx):
        del self._dists[idx]
        del self.weights[idx]
        self._update_attr()

    def _update_attr(self):
        self._p = np.array(self._weights) / sum(self.weights)

        self.model_x = rand_elements.Mixture([dist.model_x for dist in self.dists], self.weights)
        self._mode_x = self.model_x.mode

    def mode_y_x(self, x):
        return self.model_y_x(x).mode

    def _rvs(self, n, rng):
        c = rng.choice(self.n_dists, size=n, p=self._p)
        out = np.array(list(zip(np.zeros((n, *self.shape['x'])), np.zeros((n, *self.shape['y'])))),
                       dtype=[(c, self.dtype[c], self.shape[c]) for c in 'xy'])
        for i, dist in enumerate(self.dists):
            idx = np.flatnonzero(c == i)
            out[idx] = dist.rvs(size=idx.size)

        return out

    def model_y_x(self, x):
        return rand_elements.Mixture([dist.model_y_x(x) for dist in self.dists], self.weights)

    # def pf_x(self, x):
    #     # return sum(prob * dist.model_x.pf(x) for dist, prob in zip(self.dists, self._p))
    #     return self.model_x.pf(x)
    #
    # def pf_y_x(self, x):
    #     # def temp(y):
    #     #     return sum(prob * dist.model_y_x(x).pf(y) for dist, prob in zip(self.dists, self._p))
    #     # return temp
    #     return self.model_y_x(x).pf


class MixtureRVx(MixinRVx, Mixture):
    def __repr__(self):
        _str = "; ".join([f"{prob}: {dist}" for dist, prob in zip(self.dists, self._p)])
        return f"MixtureRVx({_str})"

    def _update_attr(self):
        super()._update_attr()

        self._mean_x = sum(prob * dist.mean_x for dist, prob in zip(self.dists, self._p))


class MixtureRVy(MixinRVy, Mixture):
    def __repr__(self):
        _str = "; ".join([f"{prob}: {dist}" for dist, prob in zip(self.dists, self._p)])
        return f"MixtureRVy({_str})"

    # def _mean_y_x_single(self, x):
    #     return sum(prob * dist._mean_y_x_single(x) for dist, prob in zip(self.dists, self._p))

    def mean_y_x(self, x):
        return sum(prob * dist.mean_y_x(x) for dist, prob in zip(self.dists, self._p))


class MixtureRVxy(MixtureRVx, MixtureRVy):
    def __repr__(self):
        _str = "; ".join([f"{prob}: {dist}" for dist, prob in zip(self.dists, self._p)])
        return f"MixtureRVxy({_str})"


# dists_ = [NormalRegressor(basis_y_x=(lambda x: x,), weights=(w,), cov_y_x_single=10) for w in [0, 4]]
#
# # x_ = [10, 11]
# # p_x = [.5, .5]
# # # y_ = ['a', 'b']
# # y_ = [0, 1]
# # dists_ = []
# # q = [.8, .5]
# # def getdat(t):
# #     def p_y_x_(x):
# #         return [t, 1 - t] if x == x_[0] else [1 - t, t]
# #     return p_y_x_
# # for v in q:
# #     r = DataConditional.finite_model(x_, p_x, y_, getdat(v))
# #     dists_.append(r)
#
# m = Mixture(dists_, [5, 8])
# m.rvs(10)
# m.mode_x
# # m.mean_x
#
# x_p = 10
# m.model_y_x(x_p).plot_pf()
# plt.title(f"Mode={m.mode_y_x(x_p):.3f}, Mean={m.mean_y_x(x_p):.3f}")
# # m.mode_y_x(1)
# # m.mean_y_x(1)
# pass
