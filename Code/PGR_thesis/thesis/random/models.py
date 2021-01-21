"""
SL models.
"""

from typing import Optional, Dict
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from thesis.random import elements as rand_elements

from thesis.util.base import RandomGeneratorMixin, vectorize_func, vectorize_func_dec
from thesis.util import spaces

# TODO: add marginal/conditional pf methods


class Base(RandomGeneratorMixin):
    """
    Base class for supervised learning data models.
    """
    _space: Dict[str, Optional[spaces.Base]]

    def __init__(self, rng=None):
        super().__init__(rng)
        self._space = {'x': None, 'y': None}

        self._model_x = None
        # self._model_y_x = None

        self._mode_x = None

    space = property(lambda self: self._space)

    shape = property(lambda self: {key: space.shape for key, space in self._space.items()})
    size = property(lambda self: {key: space.size for key, space in self._space.items()})
    ndim = property(lambda self: {key: space.ndim for key, space in self._space.items()})

    dtype = property(lambda self: {key: space.dtype for key, space in self._space.items()})

    # TODO: use NumPy structured dtypes to detail type and shape!?!

    @property
    def model_x(self):
        return self._model_x

    def model_y_x(self, x):
        raise Exception

    # TODO: default stats to reference `model_x` and `model_y_x` attributes?

    @property
    def mode_x(self):
        return self._mode_x

    def mode_y_x(self, x):
        return vectorize_func(self._mode_y_x_single, self.shape['x'])(x)

    def _mode_y_x_single(self, x):
        raise Exception

    def plot_mode_y_x(self, x=None, ax=None):
        return self.space['x'].plot(self.mode_y_x, x, ax)

    rvs = rand_elements.Base.rvs

    def _rvs(self, n, rng):
        d_x = np.array(self.model_x.rvs(n, rng=rng))
        # d_y = np.stack([self.model_y_x(x).rvs(rng=rng) for x in d_x])
        d_y = np.array([self.model_y_x(x).rvs(rng=rng) for x in d_x])

        return np.array(list(zip(d_x, d_y)), dtype=[(c, self.dtype[c], self.shape[c]) for c in 'xy'])


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
    space: dict
    shape: dict

    def mean_y_x(self, x):
        return vectorize_func(self._mean_y_x_single, self.shape['x'])(x)

    def _mean_y_x_single(self, x):
        raise Exception

    def plot_mean_y_x(self, x=None, ax=None):
        return self.space['x'].plot(self.mean_y_x, x, ax)

    def cov_y_x(self, x):
        return vectorize_func(self._cov_y_x_single, self.shape['x'])(x)

    def _cov_y_x_single(self, x):
        raise Exception


# class DataConditionalGeneric(Base):
#     def __new__(cls, model_x, model_y_x, rng=None):
#         is_numeric_y = isinstance(model_y_x(model_x.rvs()), rand_elements.MixinRV)
#         if isinstance(model_x, rand_elements.MixinRV):
#             if is_numeric_y:
#                 return super().__new__(DataConditionalRVxy)
#             else:
#                 return super().__new__(DataConditionalRVx)
#         else:
#             if is_numeric_y:
#                 return super().__new__(DataConditionalRVy)
#             else:
#                 return super().__new__(cls)
#
#     def __init__(self, model_x, model_y_x, rng=None):
#         super().__init__(rng)
#         self._model_x = model_x
#         self._space['x'] = self._model_x.space
#         self._update_x()
#
#         self._model_y_x_ = model_y_x
#         self._space['y'] = self.model_y_x(self._model_x.rvs()).space
#
#     @property
#     def model_x(self):
#         return self._model_x
#
#     @model_x.setter
#     def model_x(self, model_x):
#         self._model_x = model_x
#         self._update_x()
#
#     def _update_x(self):
#         self._mode_x = self._model_x.mode
#
#     def model_y_x(self, x):
#         return self.model_y_x_(x)
#
#     @property
#     def model_y_x_(self):
#         return self._model_y_x_
#
#     @model_y_x_.setter
#     def model_y_x_(self, model_y_x):
#         self._model_y_x_ = model_y_x
#
#     def _mode_y_x_single(self, x):
#         return self.model_y_x(x).mode
#
#     @classmethod
#     def from_finite(cls, dists, supp_x, p_x=None, rng=None):
#         model_x = rand_elements.Finite(supp_x, p_x)
#
#         def model_y_x(x):
#             eq_supp = np.all(x == model_x.space._vals_flat, axis=tuple(range(1, 1 + model_x.space.ndim)))
#             idx = np.flatnonzero(eq_supp).item()
#             return dists[idx]
#
#         return cls(model_x, model_y_x, rng)
#
#
# class DataConditionalRVx(MixinRVx, DataConditional):
#     def _update_x(self):
#         super()._update_x()
#         self._mean_x = self._model_x.mean
#         self._cov_x = self._model_x.cov
#
#
# class DataConditionalRVy(MixinRVy, DataConditional):
#     def _mean_y_x_single(self, x):
#         return self.model_y_x(x).mean
#
#     def _cov_y_x_single(self, x):
#         return self.model_y_x(x).cov
#
#
# class DataConditionalRVxy(DataConditionalRVy, DataConditionalRVx):
#     pass


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

class DataConditional(Base):
    def __new__(cls, dists, model_x, rng=None):
        is_numeric_y = all(isinstance(dist, rand_elements.MixinRV) for dist in dists)
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

    def __init__(self, dists, model_x, rng=None):
        super().__init__(rng)

        self._dists = list(dists)
        self._model_x = model_x

        self._space['x'] = self.model_x.space
        if not isinstance(self.space['x'], spaces.FiniteGeneric):
            raise ValueError(f"Data space must be finite.")
        elif self.space['x'].set_size != len(self.dists):
            raise ValueError(f"Data space must have {len(self.dists)} elements.")

        self._space['y'] = spaces.check_spaces(self.dists)

        self._update_attr()

    def __eq__(self, other):
        if isinstance(other, DataConditional):
            return (self.model_x == other.model_x
                    and self.dists == other.dists)
        return NotImplemented

    def __deepcopy__(self, memodict=None):
        # if memodict is None:
        #     memodict = {}
        # return type(self)(self.dists, self.model_x)
        return type(self)(deepcopy(self.dists), deepcopy(self.model_x))

    dists = property(lambda self: self._dists)
    model_x = property(lambda self: self._model_x)

    @property
    def p_x(self):
        return self.model_x.p

    @p_x.setter
    def p_x(self, val):
        self.model_x.p = val
        self._update_attr()

    def _update_attr(self):
        self._mode_x = self.model_x.mode

    def model_y_x(self, x):
        # idx = self.space['x'].values.tolist().index(x)
        idx = np.flatnonzero(np.isclose(x, self.space['x'].values)).item()
        return self.dists[idx]

    def _mode_y_x_single(self, x):
        return self.model_y_x(x).mode

    @classmethod
    def from_finite(cls, dists, supp_x, p_x=None, rng=None):
        model_x = rand_elements.Finite(supp_x, p_x)
        return cls(dists, model_x, rng)


class DataConditionalRVx(MixinRVx, DataConditional):
    def _update_attr(self):
        super()._update_attr()
        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov


class DataConditionalRVy(MixinRVy, DataConditional):
    def _mean_y_x_single(self, x):
        return self.model_y_x(x).mean

    def _cov_y_x_single(self, x):
        return self.model_y_x(x).cov


class DataConditionalRVxy(DataConditionalRVy, DataConditionalRVx):
    pass


class ClassConditional(MixinRVx, Base):
    def __init__(self, dists, model_y, rng=None):
        super().__init__(rng)

        self._dists = list(dists)
        self._model_y = model_y

        self._space['y'] = self.model_y.space
        if not (isinstance(self.space['y'], spaces.Finite) and self.space['y'].ndim == 0):
            raise ValueError
        elif self.space['y'].set_shape != (len(self.dists),):
            raise ValueError("Incorrect number of conditional distributions.")
        elif not np.issubdtype(self.space['y'].dtype, 'U'):
            raise ValueError("Space must be categorical")

        self._space['x'] = spaces.check_spaces(self.dists)

        self._update_attr()

    @classmethod
    def from_finite(cls, dists, supp_y, p_y=None, rng=None):
        model_y = rand_elements.Finite(np.array(supp_y, dtype='U').flatten(), p_y)
        return cls(dists, model_y, rng)

    dists = property(lambda self: self._dists)
    model_y = property(lambda self: self._model_y)

    def set_dist_attr(self, idx, **dist_kwargs):
        for key, val in dist_kwargs.items():
            setattr(self._dists[idx], key, val)
        self._update_attr()

    @property
    def p_y(self):
        return self.model_y.p

    @p_y.setter
    def p_y(self, val):
        self.model_y.p = val
        self._update_attr()

    def _update_attr(self):
        self._model_x = rand_elements.MixtureRV(self.dists, self.p_y)
        self._mode_x = self.model_x.mode
        self._mean_x = self.model_x.mean
        self._cov_x = self.model_x.cov

    def mode_y_x(self, x):
        temp = np.array([dist.pf(x) * p for dist, p in zip(self.dists, self.p_y)])
        return self.space['y'].values[temp.argmax(axis=0)]

    def model_y_x(self, x):
        temp = np.array([dist.pf(x) * p for dist, p in zip(self.dists, self.p_y)])
        p_y_x = temp / temp.sum()
        return rand_elements.Finite(self.space['y'].values, p_y_x)

    def model_x_y(self, y):
        idx = self.space['y'].values.tolist().index(y)
        return self.dists[idx]

    def _rvs(self, n, rng):
        d_y = np.array(self.model_y.rvs(n, rng=rng))
        # d_x = np.stack([self.model_x_y(y).rvs(rng=rng) for y in d_y])
        d_x = np.array([self.model_x_y(y).rvs(rng=rng) for y in d_y])

        return np.array(list(zip(d_x, d_y)), dtype=[(c, self.dtype[c], self.shape[c]) for c in 'xy'])


# m = ClassConditional.from_finite([rand_elements.Normal(mean) for mean in [0, 4]], ['a', 'b'])
# m.mode_y_x(np.linspace(0, 4, 20))
# qq = None


class BetaLinear(MixinRVx, MixinRVy, Base):     # TODO: DRY with NormalLinear
    def __init__(self, weights=(0.,), basis_y_x=None, alpha_y_x=2., model_x=rand_elements.Beta(), rng=None):
        super().__init__(rng)

        self._space['x'] = model_x.space
        self._space['y'] = spaces.Box((0, 1))

        self.model_x = model_x

        self.weights = weights
        self.alpha_y_x = alpha_y_x

        if basis_y_x is None:
            def power_func(i):
                return vectorize_func(lambda x: np.full(self.shape['y'], (x ** i).sum()), shape=self.shape['x'])

            self._basis_y_x = tuple(power_func(i) for i in range(len(self.weights)))
        else:
            self._basis_y_x = basis_y_x

    def __repr__(self):
        return f"BetaLinear(model_x={self.model_x}, basis_y_x={self.basis_y_x}, " \
               f"weights={self.weights}, alpha_y_x={self.alpha_y_x})"

    @property
    def basis_y_x(self):
        return self._basis_y_x

    @property
    def model_x(self):
        return self._model_x

    @model_x.setter
    def model_x(self, model_x):
        self._model_x = model_x

        self._mode_x = self._model_x.mode

        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov

    def mean_y_x(self, x):
        return sum(weight * func(x) for weight, func in zip(self.weights, self._basis_y_x))

    def cov_y_x(self, x):
        mean = self.mean_y_x(x)
        return mean * (1 - mean) / (self.alpha_y_x + 1)

    def model_y_x(self, x):
        return rand_elements.Beta.from_mean(self.mean_y_x(x), self.alpha_y_x)

    def _mode_y_x_single(self, x):
        return self.model_y_x(x).mode


class NormalLinear(MixinRVx, MixinRVy, Base):
    def __init__(self, weights=(0.,), basis_y_x=None, cov_y_x=1., model_x=rand_elements.Normal(), rng=None):
        super().__init__(rng)

        self._space['x'] = model_x.space
        self.model_x = model_x

        self.weights = weights
        self.cov_y_x_ = cov_y_x

        # self._mode_y_x_single = self._mean_y_x_single

        if basis_y_x is None:
            def power_func(i):
                return vectorize_func(lambda x: np.full(self.shape['y'], (x ** i).sum()), shape=self.shape['x'])

            self._basis_y_x = tuple(power_func(i) for i in range(len(self.weights)))
        else:
            self._basis_y_x = basis_y_x

    def __repr__(self):
        return f"NormalModel(model_x={self.model_x}, basis_y_x={self.basis_y_x}, " \
               f"weights={self.weights}, cov_y_x={self._cov_repr})"

    @property
    def basis_y_x(self):
        return self._basis_y_x

    @property
    def model_x(self):
        return self._model_x

    @model_x.setter
    def model_x(self, model_x):
        self._model_x = model_x

        self._mode_x = self._model_x.mode

        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov

    # @property
    # def weights(self):
    #     return self._weights
    #
    # @weights.setter
    # def weights(self, val):
    #     self._weights = np.array(val)

    @property
    def cov_y_x_(self):
        return self._cov_repr

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

    # def _mean_y_x_single(self, x):
    #     return sum(weight * func(x) for weight, func in zip(self.weights, self._basis_y_x))

    def mean_y_x(self, x):
        return sum(weight * func(x) for weight, func in zip(self.weights, self._basis_y_x))

    def mode_y_x(self, x):
        return self.mean_y_x(x)

    def model_y_x(self, x):
        # mean = self._mean_y_x_single(x)
        mean = self.mean_y_x(x)
        cov = self._cov_y_x_single(x)
        return rand_elements.Normal(mean, cov)


# # g = NormalLinear(basis_y_x=(lambda x: 1, lambda x: x**2,), weights=(1, 2), cov_y_x=.01,
# #                  model_x=rand_elements.Normal(4))
# g = NormalLinear(basis_y_x=(lambda x: 1, lambda x: x.sum(),), weights=(1, 2), cov_y_x=.01,
#                  model_x=rand_elements.Normal([0, 0]))
# r = g.rvs(100)
# # plt.plot(r['x'], r['y'], '.')
# g.mean_y_x(np.linspace(0, 2, 20, endpoint=True))
# qq = None


class DataEmpirical(Base):
    def __new__(cls, values, counts, space=None, rng=None):
        if space is not None:
            dtype = {c: space[c].dtype for c in 'xy'}
        else:
            _dtype = np.array(values).dtype
            dtype = {c: _dtype[c].base for c in 'xy'}

        if np.issubdtype(dtype['x'], np.number):
            if np.issubdtype(dtype['y'], np.number):
                return super().__new__(DataEmpiricalRVxy)
            else:
                return super().__new__(DataEmpiricalRVx)
        else:
            if np.issubdtype(dtype['y'], np.number):
                return super().__new__(DataEmpiricalRVy)
            else:
                return super().__new__(cls)

    def __init__(self, values, counts, space=None, rng=None):
        super().__init__(rng)

        values, counts = map(np.array, (values, counts))
        if space is None:
            dtype = np.array(values).dtype
            for c in 'xy':
                if np.issubdtype(dtype[c].base, np.number):
                    self._space[c] = spaces.Euclidean(dtype[c].shape)
                else:
                    raise NotImplementedError
        else:
            self._space = space

        self.n = 0
        self.data = self._structure_data({'x': [], 'y': []}, [])
        # self._model_x = rand_elements.DataEmpirical([], [], space=self.space['x'])
        self._init_attr()
        self.add_values(values, counts)

    def __repr__(self):
        return f"DataEmpirical(space={self.space}, n={self.n})"

    @classmethod
    def from_data(cls, d, space=None, rng=None):
        return cls(*cls._count_data(d), space, rng)

    def _init_attr(self):
        self._model_x = rand_elements.DataEmpirical([], [], space=self.space['x'])
        self._models_y_x = []

    @staticmethod
    def _count_data(d):
        return np.unique(d, return_counts=True, axis=0)

    def _structure_data(self, values, counts):
        return np.array(list(zip(values['x'], values['y'], counts)),
                        dtype=[('x', self.dtype['x'], self.shape['x']),
                               ('y', self.dtype['y'], self.shape['y']),
                               ('n', np.int,)])

    def add_data(self, d):
        self.add_values(*self._count_data(d))

    def add_values(self, values, counts):
        values, counts = map(np.array, (values, counts))
        n_new = counts.sum(dtype=np.int)
        if n_new == 0:
            return

        self.n += n_new

        idx_new = []
        for i, (value, count) in enumerate(zip(values, counts)):
            idx = np.flatnonzero(value == self.data[['x', 'y']])
            if idx.size == 1:
                self.data['n'][idx.item()] += count
            else:
                idx_new.append(i)

        if len(idx_new) > 0:
            self.data = np.concatenate((self.data, self._structure_data(values[idx_new], counts[idx_new])))

        _, idx = np.unique(values['x'], axis=0, return_index=True)
        values_x_unique = values['x'][np.sort(idx)]
        # values_x_unique = np.unique(values['x'], axis=0)
        for x_add in values_x_unique:
            idx_match = np.flatnonzero(np.all(x_add == values['x'], axis=tuple(range(1, 1 + self.ndim['x']))))
            values_y, counts_y = values['y'][idx_match], counts[idx_match]

            idx = self._model_x._get_idx(x_add)
            if idx is None:
                self._models_y_x.append(rand_elements.DataEmpirical(values_y, counts_y, self.space['y']))
            else:
                self._models_y_x[idx].add_values(values_y, counts_y)

            self._model_x.add_values([x_add], [counts[idx_match].sum()])

        self._update_attr()

    def _update_attr(self):
        self._p = self.data['n'] / self.n

    @property
    def mode_x(self):
        return self._model_x.mode

    def _get_idx_x(self, x):
        idx = np.flatnonzero(np.all(x == self.model_x.data['x'], axis=tuple(range(1, 1 + self.ndim['x']))))
        if idx.size == 1:
            return idx.item()
        elif idx.size == 0:
            return None
        else:
            raise ValueError

    def model_y_x(self, x):
        # idx = np.flatnonzero(np.all(x == self.model_x.data['x'], axis=tuple(range(1, 1 + self.ndim['x']))))
        # if idx.size == 1:
        #     return self._models_y_x[idx.item()]
        # else:
        #     # raise ValueError("No matching data for empirical distribution.")
        #     return rand_elements.DataEmpirical([], [], space=self.space['y'])

        idx = self._get_idx_x(x)
        if idx is not None:
            return self._models_y_x[idx]
        else:
            # raise ValueError("No matching data for empirical distribution.")
            return rand_elements.DataEmpirical([], [], space=self.space['y'])

    # def _mode_y_x_single(self, x):
    #     return self.model_y_x(x).mode

    def _mode_y_x_single(self, x):
        idx = self._get_idx_x(x)
        if idx is not None:
            return self._models_y_x[idx].mode
        else:
            return np.nan

    def _rvs(self, size, rng):
        return rng.choice(self.data[['x', 'y']], size, p=self._p)


class DataEmpiricalRVx(MixinRVx, DataEmpirical):
    def __repr__(self):
        return f"DataEmpiricalRVx(space={self.space}, n={self.n})"

    # def _update_attr(self):
    #     super()._update_attr()
    #
    #     self._mean_x = self._model_x.mean
    #     self._cov_x = self._model_x.cov

    @property
    def mean_x(self):
        return self._model_x.mean

    @property
    def cov_x(self):
        return self._model_x.cov


class DataEmpiricalRVy(MixinRVy, DataEmpirical):
    def __repr__(self):
        return f"DataEmpiricalRVy(space={self.space}, n={self.n})"

    # def _mean_y_x_single(self, x):
    #     return self.model_y_x(x).mean

    def _mean_y_x_single(self, x):
        idx = self._get_idx_x(x)
        if idx is not None:
            return self._models_y_x[idx].mean
        else:
            return np.nan


class DataEmpiricalRVxy(DataEmpiricalRVx, DataEmpiricalRVy):
    def __repr__(self):
        return f"DataEmpiricalRVxy(space={self.space}, n={self.n})"


# # r = ClassConditional.from_finite([rand_elements.Normal(mean) for mean in [[0, 0], [4, 4]]], ['a', 'b'])
# r = ClassConditional.from_finite([rand_elements.Finite([1, 2], [p, 1-p]) for p in (.2, .5)], ['a', 'b'])
# # r = NormalLinear(weights=[1, 1], cov_y_x=np.eye(2))
# # r = NormalLinear(weights=[1, 1], cov_y_x=1., model_x=rand_elements.Normal([0, 0]))
# e = DataEmpirical.from_data(r.rvs(5), space=r.space)
# # e.add_data(r.rvs(5))
#
# print(e)
# qq = None


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

        self._space = spaces.check_spaces(self.dists)

        self.weights = weights

    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"Mixture({_str})"

    def __deepcopy__(self, memodict=None):
        # if memodict is None:
        #     memodict = {}
        return type(self)(self.dists, self.weights, self.rng)

    dists = property(lambda self: self._dists)
    n_dists = property(lambda self: len(self._dists))

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = np.array(value)
        if len(self._weights) != self.n_dists:
            raise ValueError(f"Weights must have length {self.n_dists}.")

        self._update_attr()

    def set_dist_attr(self, idx, **dist_kwargs):      # TODO: improved implementation w/ direct self.dists access?
        for key, val in dist_kwargs.items():
            setattr(self._dists[idx], key, val)
        self._update_attr()

    # def add_dist(self, dist, weight):
    #     self._dists.append(dist)
    #     self.weights.append(weight)
    #     self._update_attr()

    def set_dist(self, idx, dist, weight):  # TODO: type check?
        self._dists[idx] = dist
        self.weights[idx] = weight
        self._update_attr()  # weights setter not invoked
        # try:
        #     self._dists[idx] = dist
        #     self.weights[idx] = weight
        #     self._update_attr()     # weights setter not invoked
        # except IndexError:
        #     self.add_dist(dist, weight)

    # def del_dist(self, idx):
    #     del self._dists[idx]
    #     del self.weights[idx]
    #     self._update_attr()

    @property
    def _idx_nonzero(self):
        return np.flatnonzero(self._weights)

    def _update_attr(self):
        self._p = self._weights / self.weights.sum()

        if self._idx_nonzero.size == 1:
            self._model_x = self._dists[self._idx_nonzero.item()].model_x
        else:
            # self._mode = self.space.argmax(self.pf)
            # args = zip(*[(dist.model_x, w) for dist, w in zip(self.dists, self.weights) if w > 0])
            args = zip(*[(self.dists[i].model_x, self.weights[i]) for i in self._idx_nonzero])
            self._model_x = rand_elements.Mixture(*args)

        # self._mode_x = self.model_x.mode

    @property
    def mode_x(self):
        return self._model_x.mode

    def _mode_y_x_single(self, x):
        return self.model_y_x(x).mode

    def model_y_x(self, x):
        _weights = self._weights_y_x(x)
        idx_nonzero = np.flatnonzero(_weights)
        if idx_nonzero.size == 1:
            return self._dists[idx_nonzero.item()].model_y_x(x)
        else:
            args = zip(*[(self.dists[i].model_y_x(x), _weights[i]) for i in idx_nonzero])
            return rand_elements.Mixture(*args)

        # args = zip(*[(dist.model_y_x(x), w) for dist, w in zip(self.dists, self._weights_y_x(x)) if w > 0])
        # return rand_elements.Mixture(*args)

    def _weights_y_x(self, x):
        # return self.weights * np.array([dist.model_x.pf(x) for dist in self.dists])
        return np.array([w * dist.model_x.pf(x) for w, dist in zip(self.weights, self.dists)])
        # return np.array([w * dist.model_x.pf(x) if w > 0. else np.zeros(x.shape)
        #                  for w, dist in zip(self.weights, self.dists)])

    def _rvs(self, n, rng):
        idx_rng = rng.choice(self.n_dists, size=n, p=self._p)
        out = np.array([tuple(np.empty(self.shape[c], self.dtype[c]) for c in 'xy') for _ in range(n)],
                       dtype=[(c, self.dtype[c], self.shape[c]) for c in 'xy'])
        for i, dist in enumerate(self.dists):
            idx = np.flatnonzero(i == idx_rng)
            if idx.size > 0:
                out[idx] = dist.rvs(size=idx.size)

        return out


class MixtureRVx(MixinRVx, Mixture):
    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"MixtureRVx({_str})"

    def _update_attr(self):
        super()._update_attr()

        # self._mean_x = sum(prob * dist.mean_x for prob, dist in zip(self._p, self.dists) if prob > 0)
        # self._mean_x = sum(self._p[i] * self.dists[i].mean_x for i in self._idx_nonzero)
        self._mean_x = np.nansum([prob * dist.mean_x for prob, dist in zip(self._p, self.dists)])


class MixtureRVy(MixinRVy, Mixture):
    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"MixtureRVy({_str})"

    def mean_y_x(self, x):
        temp = self._weights_y_x(x)
        p_y_x = temp / temp.sum(0)

        # idx_nonzero = np.flatnonzero(p_y_x)
        # return sum(p_y_x[i] * self.dists[i].mean_y_x(x) for i in idx_nonzero)

        # return sum(prob * dist.mean_y_x(x) for prob, dist in zip(p_y_x, self.dists) if prob > 0)
        temp = np.array([prob * dist.mean_y_x(x) for prob, dist in zip(p_y_x, self.dists)])
        return np.nansum(temp, axis=0)


class MixtureRVxy(MixtureRVx, MixtureRVy):
    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"MixtureRVxy({_str})"


# dists_ = [NormalLinear(basis_y_x=(lambda x: x,), weights=(w,), cov_y_x=10) for w in [0, 4]]
# # dists_ = [ClassConditional.from_finite([rand_elements.Normal(mean) for mean in [i, i+2]], ['a', 'b']) for i in (0, 4)]
# # dists_ = [ClassConditional.from_finite([rand_elements.Finite([1, 2], [p, 1-p]) for p in p_], ['a', 'b'])
# #           for p_ in [(.2, .5), (.7, .4)]]
#
# # # dists_ = [NormalLinear(basis_y_x=(lambda x: x,), weights=(2,), cov_y_x=10)]
# # dists_ = [ClassConditional.from_finite([rand_elements.Normal(mean) for mean in (0, 2)], ['a', 'b'])]
# # # dists_ = [ClassConditional.from_finite([rand_elements.Finite([1, 2], [p, 1-p]) for p in (.3, .6)], ['a', 'b'])]
# # dists_.append(DataEmpirical.from_data(dists_[0].rvs(10), dists_[0].space))
#
# m = Mixture(dists_, [5, 8])
# m.rvs(10)
#
# m.model_x.plot_pf()
# plt.title(f"Mode={m.mode_x:.3f}")
# # plt.title(f"Mode={m.mode_x:.3f}, Mean={m.mean_x:.3f}")
#
# x_p = m.model_x.rvs()
# m.model_y_x(x_p).plot_pf()
# plt.title(f"Mode={m.mode_y_x(x_p)}")
# # plt.title(f"Mode={m.mode_y_x(x_p)}, Mean={m.mean_y_x(x_p)}")
# # m.mode_y_x(np.linspace(-2, 8, 100))
# # m.mean_y_x(1)
#
# m.plot_mode_y_x()
# # m.plot_mean_y_x()
#
# qq = None
