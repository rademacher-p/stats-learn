"""
SL models.
"""

# TODO: add conditional RE object?

import math
from typing import Optional

import numpy as np

from thesis.random import elements as rand_elements
# from thesis._deprecated.RE_obj_callable import FiniteRE  # TODO: note - CALLABLE!!!!

from thesis.util.generic import RandomGeneratorMixin, vectorize_func
from thesis.util import spaces


class Base(RandomGeneratorMixin):
    """
    Base class for supervised learning data models.
    """

    def __init__(self, rng=None):
        super().__init__(rng)
        self._space = {'x': None, 'y': None}

        self._mode_x = None

    space = property(lambda self: self._space)

    shape = property(lambda self: {key: space.shape for key, space in self._space.items()})
    size = property(lambda self: {key: space.size for key, space in self._space.items()})
    ndim = property(lambda self: {key: space.ndim for key, space in self._space.items()})

    dtype = property(lambda self: {key: space.dtype for key, space in self._space.items()})

    # shape = property(lambda self: self._shape)
    # size = property(lambda self: {key: math.prod(val) for key, val in self._shape.items()})
    # ndim = property(lambda self: {key: len(val) for key, val in self._shape.items()})

    @property
    def mode_x(self):
        return self._mode_x

    def mode_y_x(self, x):
        return vectorize_func(self._mode_y_x_single, self.shape['x'])(x)

    def _mode_y_x_single(self, x):
        raise NotImplementedError
        pass

    rvs = rand_elements.Base.rvs

    def _rvs(self, size, rng):
        raise NotImplementedError("Method must be overwritten.")
        pass


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
        is_numeric_y_x = isinstance(model_y_x(model_x.rvs()), rand_elements.MixinRV)
        if isinstance(model_x, rand_elements.MixinRV):
            if is_numeric_y_x:
                return super().__new__(DataConditionalRVxy)
            else:
                return super().__new__(DataConditionalRVx)
        else:
            if is_numeric_y_x:
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

    def _rvs(self, n, rng):
        d_x = np.array(self.model_x.rvs(n, rng=rng))
        d_y = np.stack([self.model_y_x(x).rvs(rng=rng) for x in d_x])

        d = np.array(list(zip(d_x, d_y)), dtype=[('x', d_x.dtype, self.shape['x']), ('y', d_y.dtype, self.shape['y'])])

        return d

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

    def _mean_y_x_single(self, x):
        return self._model_y_x(x).mean

    def _cov_y_x_single(self, x):
        return self._model_y_x(x).cov


class DataConditionalRVxy(MixinRVy, DataConditionalRVx):
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


class NormalRegressor(MixinRVx, MixinRVy, Base):        # TODO: rename NormalLinear?
    # param_names = ('model_x', 'basis_y_x', 'cov_y_x_single', 'weights')

    def __init__(self, weights=(0.,), basis_y_x=None, cov_y_x_single=1., model_x=rand_elements.Normal(), rng=None):
        super().__init__(rng)

        self.model_x = model_x
        self.weights = weights
        self.cov_y_x_single = cov_y_x_single         # TODO: override inherited, change attr name?

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
    def cov_y_x_single(self):
        return self._cov_y_x_single

    @cov_y_x_single.setter
    def cov_y_x_single(self, val):
        if callable(val):
            self._cov_repr = val
            self._cov_y_x_single = self._cov_repr
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

    # _rvs = DataConditional._rvs
    def _rvs(self, n, rng):
        d_x = np.array(self.model_x.rvs(n, rng=rng))
        d_y = np.stack([self.model_y_x(x).rvs(rng=rng) for x in d_x])

        d = np.array(list(zip(d_x, d_y)), dtype=[('x', d_x.dtype, self.shape['x']), ('y', d_y.dtype, self.shape['y'])])

        return d

# g = NormalRegressor(basis_y_x=(lambda x: x,), weights=(1,), cov_y_x_single=.01)
# r = g.rvs(100)
# # plt.plot(r['x'], r['y'], '.')


class GenericEmpirical(Base):

    # FIXME FIXME: import structured array with additional counts field from `random.elements` !!!

    def __new__(cls, d, space=None, rng=None):
        dtype = np.array(d).dtype
        if all(np.issubdtype(dtype[c].base, np.number) for c in 'xy'):
            return super().__new__(GenericEmpiricalRVxy)
        else:
            return super().__new__(cls)

    def __init__(self, d, space=None, rng=None):
        super().__init__(rng)

        if space is None:
            self._space = spaces.Euclidean(d[0].shape)
        else:
            self._space = space

        self.n = len(d)
        self.values, self.counts = self._count_data(d)
        self._values_flat = self._struct_flatten(self.values)

        self._update_attr()

    def _struct_flatten(self, val):
        x, y = val['x'].reshape(-1, self.size['x']), val['y'].reshape(-1, self.size['y'])
        return np.array(list(zip(x, y)), dtype=[('x', self.dtype['x'], (self.size['x'],)),
                                                ('y', self.dtype['y'], (self.size['y'],))])

    def __repr__(self):
        return f"GenericEmpirical(space={self.space}, n={self.n})"

    def _update_attr(self):
        self.p = self.counts / self.n
        # self._mode = self.values[self.counts.argmax()]

        # self.values_x, _idx = np.unique(self.values['x'], return_inverse=True, axis=0)
        # self.counts_x = np.array([self.counts[_idx == i].sum() for i in range(len(self.values_x))])

        self.values_x, _idx = np.unique(self.values['x'], return_inverse=True, axis=0)
        self.counts_x = np.empty(len(self.values_x), dtype=np.int)
        self.values_y_x, self.counts_y_x = [], []
        for i in range(len(self.values_x)):
            values_, counts_ = self.values['y'][_idx == i], self.counts[_idx == i]
            self.counts_x[i] = counts_.sum()

            self.values_y_x.append(values_)
            self.counts_y_x.append(counts_)

        self._mode_x = self.values_x[self.counts_x.argmax()]

        # TODO: efficient with setattr?
        # self.values_y, _idx = np.unique(self.values['y'], return_inverse=True, axis=0)
        # self.counts_y = np.array([self.counts[_idx == i].sum() for i in range(len(self.values_y))])
        # self._mode_y = self.values_y[self.counts_y.argmax()]

    def info_y_x(self, x):
        eq = np.all(x.flatten() == self.values_x.reshape(-1, self.size['x']), axis=-1)
        if eq.sum() == 1:
            idx = np.flatnonzero(eq).item()
            return self.values_y_x[idx], self.counts_y_x[idx]
        else:
            return None, None

    def _mode_y_x_single(self, x):
        values, counts = self.info_y_x(x)
        if values is not None:
            return values[counts.argmax()]
        else:
            return np.nan     # TODO: value?

    @staticmethod
    def _count_data(d):
        return np.unique(d, return_counts=True, axis=0)

    def add_data(self, d):
        self.n += len(d)

        values_new, counts_new = [], []
        for value, count in zip(*self._count_data(d)):
            eq = self._struct_flatten(value) == self._values_flat
            if eq.sum() == 1:
                self.counts[eq] += count
            elif eq.sum() == 0:
                values_new.append(value)
                counts_new.append(count)
            else:
                raise ValueError

        if len(values_new) > 0:
            values_new, counts_new = np.array(values_new), np.array(counts_new)
            self.values = np.concatenate((self.values, values_new))
            self.counts = np.concatenate((self.counts, counts_new))

            self._values_flat = np.concatenate((self._values_flat, self._struct_flatten(values_new)))

        self._update_attr()        # TODO: add efficient updates

    def _rvs(self, size, rng):
        return rng.choice(self.values, size, p=self.p)


class GenericEmpiricalRVxy(MixinRVx, MixinRVy, GenericEmpirical):
    def __repr__(self):
        return f"GenericEmpiricalRVxy(space={self.space}, n={self.n})"

    def _update_attr(self):
        super()._update_attr()

        mean_flat = (self._values_flat * self.p[:, np.newaxis]).sum(0)
        self._mean = mean_flat.reshape(self.shape)

        ctr_flat = self._values_flat - mean_flat
        outer_flat = (ctr_flat[:, np.newaxis] * ctr_flat[..., np.newaxis]).reshape(len(ctr_flat), -1)
        self._cov = (self.p[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self.shape)


# x_ = [10, 11]
# def p_y_x_(x):
#     return [.8, .2] if x == 0 else [.2, .8]
# r = DataConditional.finite_model(x_, [.5, .5], ['a', 'b'], p_y_x_)

r = NormalRegressor(weights=[1, 1], cov_y_x_single=np.eye(2))
# r = NormalRegressor(weights=[1, 1], cov_y_x_single=1., model_x=rand_elements.Normal([0, 0]))

d_ = r.rvs(10)
e = GenericEmpirical(d_, space=r.space)
e.add_data(r.rvs(5))
print(e)
pass
