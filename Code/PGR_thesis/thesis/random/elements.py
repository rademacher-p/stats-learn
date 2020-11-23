"""
Random elements.
"""

# TODO: docstrings?
# TODO: do ABC or PyCharm bug?

import math
from typing import Optional
from numbers import Integral

import numpy as np
from scipy.stats._multivariate import _PSD
from scipy.special import gammaln, xlogy, xlog1py, betaln
import matplotlib.pyplot as plt

from thesis.util.base import RandomGeneratorMixin, check_data_shape, check_valid_pmf, vectorize_func
from thesis.util import plotting, spaces


#%% Base RE classes

class Base(RandomGeneratorMixin):
    """
    Base class for random element objects.
    """

    def __init__(self, rng=None):
        super().__init__(rng)

        self._space = None      # TODO: arg?

        self._mode = None       # TODO: make getter do numerical approx if None!?

    space = property(lambda self: self._space)

    shape = property(lambda self: self._space.shape)
    size = property(lambda self: self._space.size)
    ndim = property(lambda self: self._space.ndim)

    dtype = property(lambda self: self._space.dtype)

    mode = property(lambda self: self._mode)

    def pf(self, x):    # TODO: perform input checks using `space.__contains__`?
        return vectorize_func(self._pf_single, self.shape)(x)     # TODO: decorator? better way?

    def _pf_single(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass

    def plot_pf(self, x=None, ax=None):
        return self.space.plot(self.pf, x, ax)

    def rvs(self, size=None, rng=None):
        if size is None:
            shape = ()
        elif isinstance(size, (Integral, np.integer)):
            shape = (size,)
        elif isinstance(size, tuple):
            shape = size
        else:
            raise TypeError("Input 'size' must be int or tuple.")

        rng = self._get_rng(rng)
        # return self._rvs(math.prod(shape), rng).reshape(shape + self.shape)
        rvs = self._rvs(math.prod(shape), rng)
        return rvs.reshape(shape + rvs.shape[1:])    # TODO: use np.asscalar if possible?

    def _rvs(self, n, rng):
        raise NotImplementedError("Method must be overwritten.")
        pass


class MixinRV:
    _mean: Optional[np.ndarray]
    _cov: Optional[np.ndarray]

    # mean = property(lambda self: self._mean)
    # cov = property(lambda self: self._cov)

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov


class BaseRV(MixinRV, Base):
    """
    Base class for random variable (numeric) objects.
    """

    def __init__(self, rng=None):
        super().__init__(rng)
        # self._mean = None
        # self._cov = None


#%% Specific RE's

class Deterministic(Base):
    """
    Deterministic random element.
    """

    # TODO: redundant, just use Finite? or change to ContinuousRV for integration? General dirac mix?

    def __new__(cls, val, rng=None):
        if np.issubdtype(np.array(val).dtype, np.number):
            return super().__new__(DeterministicRV)
        else:
            return super().__new__(cls)

    def __init__(self, val, rng=None):
        super().__init__(rng)
        self.val = val

    # Input properties
    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = np.array(val)
        self._space = spaces.FiniteGeneric(self._val, shape=self._val.shape)

        self._mode = self._val

    def _rvs(self, n, rng):
        return np.broadcast_to(self._val, (n, *self.shape))

    def pf(self, x):
        return np.where(np.all(x.reshape(-1, self.size) == self._val.flatten(), axis=-1), 1., 0.)


class DeterministicRV(MixinRV, Deterministic):
    """
    Deterministic random variable.
    """

    # @property
    # def val(self):
    #     return self.val

    @Deterministic.val.setter
    # @val.setter
    def val(self, val):
        # super(DeterministicRV, self.__class__).val.fset(self, val)
        Deterministic.val.fset(self, val)

        self._mean = self._val
        self._cov = np.zeros(2 * self.shape)


# rng = np.random.default_rng()
# a = np.arange(6).reshape(3, 2)
# # a = ['a','b','c']
# b = Deterministic(a[1], rng)
# b.mode
# b.mean
# b.cov
# b.pf(b.rvs(8))


# TODO: rename generic?
class Finite(Base):     # TODO: DRY - use stat approx from the Finite space's methods?
    """
    Generic RE drawn from a finite support set using an explicitly defined PMF.
    """

    def __new__(cls, supp, p=None, rng=None):
        if np.issubdtype(np.array(supp).dtype, np.number):
            return super().__new__(FiniteRV)
        else:
            return super().__new__(cls)

    def __init__(self, supp, p=None, rng=None):
        super().__init__(rng)

        _supp = np.array(supp)

        if p is None:
            size_p = _supp.shape[0]
            p = np.ones(size_p) / size_p
        else:
            p = np.array(p)

        self._space = spaces.FiniteGeneric(_supp, shape=_supp.shape[p.ndim:])

        self.p = p
        # self._update_attr()

    def __repr__(self):
        return f"FiniteRE(support={self.supp}, p={self.p})"

    # Input properties
    @property
    def supp(self):
        return self.space.values

    @property
    def _supp_flat(self):
        return self.space._vals_flat

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = check_valid_pmf(p)
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        if self._p.shape != self.space.set_shape:
            raise ValueError(f"Shape of 'p' must be {self.space.set_shape}.")
        self._p_flat = self._p.flatten()

        self._mode = self._supp_flat[np.argmax(self._p_flat)]

    def _rvs(self, n, rng):
        return rng.choice(self._supp_flat, size=n, p=self._p_flat)

    def _pf_single(self, x):
        eq_supp = np.all(x == self._supp_flat, axis=tuple(range(1, 1 + self.ndim)))
        if eq_supp.sum() == 0:
            raise ValueError("Input 'x' must be in the support.")

        return self._p_flat[eq_supp].squeeze()


class FiniteRV(MixinRV, Finite):
    """
    Generic RV drawn from a finite support set using an explicitly defined PMF.
    """

    def _update_attr(self):
        super()._update_attr()

        # mean_flat = self._p_flat @ self._supp_flat
        # self._mean = mean_flat.reshape(self.shape)
        #
        # ctr_flat = self._supp_flat - mean_flat
        # outer_flat = (ctr_flat[:, np.newaxis] * ctr_flat[..., np.newaxis]).reshape(self._p.size, -1)
        # self._cov = (self._p_flat @ outer_flat).reshape(2 * self.shape)

        # TODO: clean up?

        self._mean = np.tensordot(self._p_flat, self._supp_flat, axes=[0, 0])

        ctr = self._supp_flat - self._mean
        # outer = np.empty((len(ctr), *(2 * self.shape)))
        # for i, ctr_i in enumerate(ctr):
        #     outer[i] = np.tensordot(ctr_i, ctr_i, 0)
        # self._cov = np.tensordot(self._p, outer, axes=[0, 0])
        self._cov = sum(p_i * np.tensordot(ctr_i, ctr_i, 0) for p_i, ctr_i in zip(self._p_flat, ctr))


# s = np.random.random((3, 1, 2))
# pp = np.random.random((3,))
# pp = pp / pp.sum()
# f = Finite(s, pp)
# f.rvs((2, 3))
# f.pf(f.rvs((4, 5)))
#
# s = plotting.mesh_grid([0, 1], [0, 1, 2])
# p_ = np.random.random((2, 3))
# p_ = p_ / p_.sum()
# # s, p_ = ['a', 'b', 'c'], [.3, .2, .5]
# f2 = Finite(s, p_)
# f2.pf(f2.rvs(4))
# f2.plot_pf()
# qq = None


def _dirichlet_check_input(x, alpha_0, mean):
    x, set_shape = check_valid_pmf(x, data_shape=mean.shape)

    if np.logical_and(x == 0, mean < 1 / alpha_0).any():
        raise ValueError("Each element in 'x' must be greater than "
                         "zero if the corresponding mean element is less than 1 / alpha_0.")

    return x, set_shape


class Dirichlet(BaseRV):
    """
    Dirichlet random process, finite-supp realizations.
    """

    def __init__(self, alpha_0, mean, rng=None):
        super().__init__(rng)
        self._space = spaces.Simplex(np.array(mean).shape)

        self._alpha_0 = alpha_0
        self._mean = check_valid_pmf(mean, full_support=True)
        self._update_attr()

    # Input properties
    @property
    def alpha_0(self):
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = alpha_0
        self._update_attr()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = check_valid_pmf(mean, full_support=True)
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        if np.min(self._mean) > 1 / self._alpha_0:
            self._mode = (self._mean - 1 / self._alpha_0) / (1 - self.size / self._alpha_0)
        else:
            # warnings.warn("Mode method currently supported for mean > 1/alpha_0 only")
            self._mode = None       # TODO: complete with general formula

        self._cov = (np.diagflat(self._mean).reshape(2 * self.shape)
                     - np.tensordot(self._mean, self._mean, 0)) / (self._alpha_0 + 1)

        self._log_pf_coef = gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))

        self._set_x_plot()

    def _rvs(self, n, rng):
        return rng.dirichlet(self._alpha_0 * self._mean.flatten(), size=n).reshape(n, *self.shape)

    def pf(self, x):
        x, set_shape = _dirichlet_check_input(x, self._alpha_0, self._mean)

        log_pf = self._log_pf_coef + np.sum(xlogy(self._alpha_0 * self._mean - 1, x).reshape(-1, self.size), -1)
        return np.exp(log_pf).reshape(set_shape)

    def _set_x_plot(self):
        x = plotting.simplex_grid(30, self.shape, hull_mask=(self.mean < 1 / self.alpha_0))
        self._space.set_x_plot(x)


# rng_ = np.random.default_rng()
# a0 = 10
# m = np.random.random(3)
# m = m / m.sum()
# d = Dirichlet(a0, m, rng_)
# d.plot_pf()
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pf(d.rvs())
# d.pf(d.rvs(4).reshape((2, 2)+d.mean.shape))


def _empirical_check_input(x, n, mean):
    x, set_shape = check_valid_pmf(x, data_shape=mean.shape)

    if (np.minimum((n * x) % 1, (-n * x) % 1) > 1e-9).any():
        raise ValueError("Each entry in 'x' must be a multiple of 1/n.")

    return x, set_shape


class Empirical(BaseRV):
    """
    Empirical random process, finite-supp realizations.
    """

    def __init__(self, n, mean, rng=None):
        super().__init__(rng)
        self._space = spaces.SimplexDiscrete(self._n, np.array(mean).shape)

        self._n = n
        self._mean = check_valid_pmf(mean)
        self._update_attr()

    # Input properties
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n
        self._update_attr()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = check_valid_pmf(mean)
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        self._log_pf_coef = gammaln(self._n + 1)

        # self._mode = ((self._n * self._mean) // 1) + simplex_round((self._n * self._mean) % 1)  # FIXME: broken
        self._mode = None

        self._cov = (np.diagflat(self._mean).reshape(2 * self.shape)
                     - np.tensordot(self._mean, self._mean, 0)) / self._n

    def _rvs(self, n, rng):
        return rng.multinomial(self._n, self._mean.flatten(), size=n).reshape(n, *self.shape) / self._n

    def pf(self, x):
        x, set_shape = _empirical_check_input(x, self._n, self._mean)

        log_pf = self._log_pf_coef + (xlogy(self._n * x, self._mean)
                                      - gammaln(self._n * x + 1)).reshape(-1, self.size).sum(axis=-1)
        return np.exp(log_pf).reshape(set_shape)


# rng = np.random.default_rng()
# n = 10
# # m = np.random.random((1, 3))
# m = np.random.default_rng().integers(10, size=(3,))
# m = m / m.sum()
# d = Empirical(n, m, rng)
# d.plot_pf()
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pf(d.rvs())
# d.pf(d.rvs(4).reshape((2, 2) + d.mean.shape))


class DirichletEmpirical(BaseRV):
    """
    Dirichlet-Empirical random process, finite-supp realizations.
    """

    def __init__(self, n, alpha_0, mean, rng=None):
        super().__init__(rng)
        self._space = spaces.SimplexDiscrete(n, np.array(mean).shape)

        self._n = n
        self._alpha_0 = alpha_0
        self._mean = check_valid_pmf(mean)
        self._update_attr()

    # Input properties
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n
        self._update_attr()

    @property
    def alpha_0(self):
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = alpha_0
        self._update_attr()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = check_valid_pmf(mean)
        if self._mean.shape != self.shape:
            raise ValueError(f"Mean shape must be {self.shape}.")
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        # TODO: mode?

        self._cov = ((self._n + self._alpha_0) / self._n / (1 + self._alpha_0)
                     * (np.diagflat(self._mean).reshape(2 * self.shape) - np.tensordot(self._mean, self._mean, 0)))

        self._log_pf_coef = (gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))
                             + gammaln(self._n + 1) - gammaln(self._alpha_0 + self._n))

    def _rvs(self, n, rng):
        raise NotImplementedError

    def pf(self, x):
        x, set_shape = _empirical_check_input(x, self._n, self._mean)

        log_pf = self._log_pf_coef + (gammaln(self._alpha_0 * self._mean + self._n * x) - gammaln(self._n * x + 1))\
            .reshape(-1, self.size).sum(axis=-1)
        return np.exp(log_pf).reshape(set_shape)


# rng_ = np.random.default_rng()
# n = 10
# a0 = 600
# m = np.ones((3,))
# m = m / m.sum()
# d = DirichletEmpirical(n, a0, m, rng_)
# d.plot_pf()
# d.mean
# d.mode
# d.cov


class Beta(BaseRV):
    """
    Beta random variable.
    """

    def __init__(self, a, b, rng=None):
        super().__init__(rng)
        self._space = spaces.Box((0, 1))

        if a <= 0 or b <= 0:
            raise ValueError("Parameters must be strictly positive.")
        self._a, self._b = a, b

        self._update_attr()

    def __repr__(self):
        return f"Beta({self.a}, {self.b})"

    # Input properties
    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        if a <= 0:
            raise ValueError
        self._a = a
        self._update_attr()

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        if b <= 0:
            raise ValueError
        self._b = b
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        if self._a > 1:
            if self._b > 1:
                self._mode = (self._a - 1) / (self._a + self._b - 2)
            else:
                self._mode = 1
        elif self._a <= 1:
            if self._b > 1:
                self._mode = 0
            elif self._a == 1 and self._b == 1:
                self._mode = 0      # any in unit interval
            else:
                self._mode = 0      # any in {0,1}

        self._mean = self._a / (self._a + self._b)
        self._cov = self._a * self._b / (self._a + self._b)**2 / (self._a + self._b + 1)

    def _rvs(self, n, rng):
        return rng.beta(self._a, self._b, size=n)

    def pf(self, x):
        x = np.array(x)
        log_pf = xlog1py(self._b - 1.0, -x) + xlogy(self._a - 1.0, x) - betaln(self._a, self._b)
        return np.exp(log_pf)


class Normal(BaseRV):
    def __init__(self, mean=0., cov=1., rng=None):
        """
        Normal random variable.

        Parameters
        ----------
        mean : float or Iterable of float
            Mean
        cov : float or np.ndarray
            Covariance
        rng : np.random.Generator or int, optional
            Random number generator

        """

        super().__init__(rng)
        self._space = spaces.Euclidean(np.array(mean).shape)

        self.mean = mean
        self.cov = cov

    def __repr__(self):
        return f"Normal(mean={self.mean}, cov={self.cov})"

    @property
    def mean(self):
        return self._mean

    @mean.setter
    # @BaseRV.mean.setter
    def mean(self, mean):
        self._mean = np.array(mean)
        if self._mean.shape != self.shape:
            raise ValueError(f"Mean array shape must be {self.shape}.")
        self._mean_flat = self._mean.flatten()

        self._mode = self._mean

        try:
            self._set_x_plot()
        except AttributeError:      # TODO: workaround for init - wack?
            pass

    @property
    def cov(self):
        return self._cov

    @cov.setter
    # @BaseRV.cov.setter
    def cov(self, cov):
        self._cov = np.array(cov)

        if self._cov.shape == () and self.ndim == 1:    # FIXME: hack-ish?
            self._cov = self._cov * np.eye(self.size)

        if self._cov.shape != self.shape * 2:
            raise ValueError(f"Covariance array shape must be {self.shape * 2}.")
        self._cov_flat = self._cov.reshape(2 * (self.size,))

        psd = _PSD(self._cov_flat, allow_singular=False)
        self.prec_U = psd.U
        self._log_pf_coef = -0.5 * (psd.rank * np.log(2 * np.pi) + psd.log_pdet)

        self._set_x_plot()

    def _rvs(self, n, rng):
        return rng.multivariate_normal(self._mean_flat, self._cov_flat, size=n).reshape(n, *self.shape)

    def pf(self, x):
        x, set_shape = check_data_shape(x, self.shape)

        dev = x.reshape(-1, self.size) - self._mean_flat
        maha = np.sum(np.square(np.dot(dev, self.prec_U)), axis=-1)

        log_pf = self._log_pf_coef + -0.5 * maha.reshape(set_shape)
        return np.exp(log_pf)

    def _set_x_plot(self):
        if self.shape in {(), (2,)}:
            if self.shape == ():
                lims = self._mean.item() + np.array([-1, 1]) * 3 * np.sqrt(self._cov.item())
            else:   # self.shape == (2,):
                lims = [(self._mean[i] - 3 * np.sqrt(self._cov[i, i]), self._mean[i] + 3 * np.sqrt(self._cov[i, i]))
                        for i in range(2)]

            self._space.lims_plot = lims

# # mean_, cov_ = 1., 1.
# mean_, cov_ = np.ones(2), np.eye(2)
# norm = Normal(mean_, cov_)
# norm.rvs(5)
# plt_data = norm.plot_pf()
#
# delta = 0.01
# # x = np.arange(-4, 4, delta)
# x = np.stack(np.meshgrid(np.arange(-4, 4, delta), np.arange(-4, 4, delta)), axis=-1)
#
# y = norm.pf(x)
# print(delta**2*y.sum())


class NormalLinear(Normal):     # TODO: rework, only allow weights and cov to be set?
    def __init__(self, weights=(0.,), basis=np.ones(1), cov=(1.,), rng=None):
        # self._set_weights(weights)
        # self._set_basis(basis)
        self._basis = np.array(basis)

        _mean_temp = np.empty(self._basis.shape[:-1])
        super().__init__(_mean_temp, cov, rng)

        self.weights = weights

    def __repr__(self):
        return f"NormalLinear(weights={self.weights}, basis={self.basis}, cov={self.cov})"

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, val):
        # self._set_weights(val)
        # self._set_mean()
        self._weights = np.array(val)
        if self._weights.ndim != 1:
            raise ValueError
        self.mean = self._basis @ self._weights

    # def _set_weights(self, val):
    #     self._weights = np.array(val)
    #     if self._weights.ndim != 1:
    #         raise ValueError

    @property
    def basis(self):
        return self._basis

    # @basis.setter
    # def basis(self, val):
    #     self._set_basis(val)
    #     self._set_mean()

    # def _set_basis(self, val):
    #     self._basis = np.array(val)
    #     self._shape = self._basis.shape[:-1]

    # def _set_mean(self):
    #     self.mean = self._basis @ self._weights


# bs = [[1, 0], [0, 1], [1, 1]]
# a = NormalLinear(weights=np.ones(2), basis=np.array(bs), cov=np.eye(3))


class DataEmpirical(Base):

    # TODO: subclass for FiniteGeneric space?
    # TODO: subclass for continuous spaces, easy stats?

    def __new__(cls, values, counts, space=None, rng=None):
        if np.issubdtype(np.array(values).dtype, np.number):
            return super().__new__(DataEmpiricalRV)
        else:
            return super().__new__(cls)

    def __init__(self, values, counts, space=None, rng=None):
        super().__init__(rng)

        values, counts = np.array(values), np.array(counts)
        if space is None:
            if np.issubdtype(values.dtype, np.number):
                self._space = spaces.Euclidean(values.shape[1:])
            else:
                raise NotImplementedError
        else:
            self._space = space

        self.n = counts.sum()
        self.data = self._structure_data(values, counts)

        self._update_attr()

    def __repr__(self):
        return f"DataEmpirical(space={self.space}, n={self.n})"

    @classmethod
    def from_data(cls, d, space=None, rng=None):
        return cls(*cls._count_data(d), space, rng)

    @staticmethod
    def _count_data(d):
        return np.unique(d, return_counts=True, axis=0)

    def _structure_data(self, values, counts):
        return np.array(list(zip(values, counts)), dtype=[('x', self.dtype, self.shape), ('n', np.int,)])

    def _get_idx(self, value):
        idx = np.flatnonzero(np.all(value == self.data['x'], axis=tuple(range(1, 1 + self.ndim))))
        if idx.size == 1:
            return idx.item()
        elif idx.size == 0:
            return None
        else:
            raise ValueError

    def add_values(self, values, counts):
        values, counts = np.array(values), np.array(counts)
        self.n += counts.sum()

        idx_new = []
        for i, (value, count) in enumerate(zip(values, counts)):
            idx = self._get_idx(value)
            if idx is not None:
                self.data['n'][idx] += count
            else:
                idx_new.append(i)

        if len(idx_new) > 0:
            self.data = np.concatenate((self.data, self._structure_data(values[idx_new], counts[idx_new])))

        self._update_attr()

    def add_data(self, d):
        self.add_values(*self._count_data(d))

    def _update_attr(self):
        self._p = self.data['n'] / self.n
        self._mode = self.data['x'][self.data['n'].argmax()]

        self._set_x_plot()

    def _rvs(self, size, rng):
        return rng.choice(self.data['x'], size, p=self._p)

    def _pf_single(self, x):
        # if x not in self.space:
        #     raise ValueError("Input 'x' must be in the support.")

        idx = self._get_idx(x)
        if idx is not None:
            return self._p[idx]     # TODO: implement infinite valued output for continuous space!? use CDF?!
        else:
            return 0.

    def _set_x_plot(self):
        if isinstance(self.space, spaces.Continuous) and self.shape in {()}:
            # TODO: hackish? adds empirical values to the plot support (so impulses are not missed)
            x = np.sort(np.unique(np.concatenate((self.space.x_plt, self.data['x']))))
            self._space.set_x_plot(x)


class DataEmpiricalRV(MixinRV, DataEmpirical):
    def __repr__(self):
        return f"DataEmpiricalRV(space={self.space}, n={self.n})"

    def _update_attr(self):
        super()._update_attr()

        self._mean = np.tensordot(self._p, self.data['x'], axes=[0, 0])

        ctr = self.data['x'] - self._mean
        # outer = np.empty((len(ctr), *(2 * self.shape)))
        # for i, ctr_i in enumerate(ctr):
        #     outer[i] = np.tensordot(ctr_i, ctr_i, 0)        # TODO: try np.einsum?
        # self._cov = np.tensordot(self._p, outer, axes=[0, 0])
        self._cov = sum(p_i * np.tensordot(ctr_i, ctr_i, 0) for p_i, ctr_i in zip(self._p, ctr))


# # r = Beta(5, 5)
# # r = Finite(plotting.mesh_grid([0, 1], [3, 4, 5]), np.ones((2, 3)) / 6)
# # r = Finite(['a', 'b'], [.6, .4])
# # e = DataEmpirical.from_data(r.rvs(10), space=r.space)
# # e.add_data(r.rvs(5))
#
# e = DataEmpirical(['a', 'b'], [5, 6], space=spaces.FiniteGeneric(['a', 'b']))
# e.add_values(['b', 'c'], [4, 1])
#
# print(e)
# e.plot_pf()
# qq = None


class Mixture(Base):
    def __new__(cls, dists, weights, rng=None):
        if all(isinstance(dist, MixinRV) for dist in dists):
            return super().__new__(MixtureRV)
        else:
            return super().__new__(cls)

    def __init__(self, dists, weights, rng=None):       # TODO: special implementation for Finite? get modes, etc?
        super().__init__(rng)
        self._dists = list(dists)

        self._space = spaces.check_spaces(self.dists)
        # self._space = self.dists[0].space
        # if not all(dist.space == self.space for dist in self.dists[1:]):
        #     raise ValueError("All distributions must have the same space.")

        self.weights = weights

    def __repr__(self):
        _str = "; ".join([f"{prob}: {dist}" for dist, prob in zip(self.dists, self._p)])
        return f"Mixture({_str})"

    dists = property(lambda self: self._dists)
    n_dists = property(lambda self: len(self._dists))

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
        dist = self._dists[idx]
        for key, val in dist_kwargs.items():
            setattr(dist, key, val)
        self._update_attr()

    def add_dist(self, dist, weight):       # TODO: type check?
        self._dists.append(dist)
        self.weights.append(weight)
        self._update_attr()

    def set_dist(self, idx, dist, weight):
        try:
            self._dists[idx] = dist
            self.weights[idx] = weight
            self._update_attr()     # weights setter not invoked
        except IndexError:
            self.add_dist(dist, weight)

    def del_dist(self, idx):
        del self._dists[idx]
        del self.weights[idx]
        self._update_attr()

    def _update_attr(self):
        self._set_x_plot()

        self._p = np.array(self._weights) / sum(self.weights)
        self._mode = self.space.argmax(self.pf)

    def _rvs(self, n, rng):
        idx_rng = rng.choice(self.n_dists, size=n, p=self._p)
        out = np.empty((n, *self.shape), dtype=self.space.dtype)
        for i, dist in enumerate(self.dists):
            idx = np.flatnonzero(i == idx_rng)
            if idx.size > 0:
                out[idx] = dist.rvs(size=idx.size)

        return out

    def pf(self, x):
        return sum(prob * dist.pf(x) for dist, prob in zip(self.dists, self._p))

    def _set_x_plot(self):
        if isinstance(self.space, spaces.Euclidean):
            temp = np.stack(list(dist.space.lims_plot for dist in self.dists))
            self._space.lims_plot = [temp[:, 0].min(), temp[:, 1].max()]

        if isinstance(self.space, spaces.Continuous) and self.shape in {()}:
            x = self.space.x_plt
            for dist in self.dists:
                if isinstance(dist, DataEmpirical):
                    x = np.concatenate((x, dist.data['x']))

            x = np.sort(np.unique(x))
            self._space.set_x_plot(x)


class MixtureRV(MixinRV, Mixture):
    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for dist, w in zip(self.dists, self.weights)])
        return f"MixtureRV({_str})"

    def _update_attr(self):
        super()._update_attr()

        self._mean = sum(prob * dist.mean for dist, prob in zip(self.dists, self._p))
        self._cov = None  # TODO: numeric approx from `space`?


# # dists_ = [Beta(*args) for args in [[10, 5], [2, 12]]]
# # dists_ = [Normal(mean, 1) for mean in [0, 4]]
# # dists_ = [Normal(mean, 1) for mean in [[0, 0], [2, 3]]]
# # dists_ = [Finite(['a', 'b'], p=[p_, 1-p_]) for p_ in [0, 1]]
# # dists_ = [Finite([[0, 0], [0, 1]], p=[p_, 1-p_]) for p_ in [0, 1]]
# dists_ = [Normal(), DataEmpirical(Normal().rvs(10))]
#
# m = Mixture(dists_, [5, 8])
# m.rvs(10)
# m.plot_pf()
# print(m.space.integrate(m.pf))
# print(m.space.moment(m.pf))
# pass
