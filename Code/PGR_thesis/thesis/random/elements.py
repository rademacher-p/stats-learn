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

from thesis.util.generic import RandomGeneratorMixin, check_data_shape, check_valid_pmf, vectorize_func
from thesis.util.math import outer_gen, diag_gen, simplex_round
from thesis.util import plot, spaces


#%% Base RE classes

class Base(RandomGeneratorMixin):
    """
    Base class for generic random element objects.
    """

    def __init__(self, rng=None):
        super().__init__(rng)

        self._space = None      # TODO: arg?

        self._mode = None

    space = property(lambda self: self._space)

    shape = property(lambda self: self._space.shape)
    size = property(lambda self: self._space.size)
    ndim = property(lambda self: self._space.ndim)

    mode = property(lambda self: self._mode)

    def pf(self, x):
        return vectorize_func(self._pf_single, self.shape)(x)     # TODO: decorator? better way?

    def _pf_single(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass

    # def plot_pf(self, ax=None):
    #     raise NotImplementedError
    #     pass

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
        return self._rvs(math.prod(shape), rng).reshape(shape + self.shape)
        # rvs = self._rvs(math.prod(shape), rng)
        # return rvs.reshape(shape + rvs.shape[1:])     # FIXME

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
    Base class for generic random variable (numeric) objects.
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


class Finite(Base):
    """
    Generic RE drawn from a finite support set using an explicitly defined PMF.
    """

    def __new__(cls, supp, p, rng=None):
        if np.issubdtype(np.array(supp).dtype, np.number):
            return super().__new__(FiniteRV)
        else:
            return super().__new__(cls)

    def __init__(self, supp, p, rng=None):
        super().__init__(rng)

        _supp = np.array(supp)
        self._p = check_valid_pmf(p)
        self._space = spaces.FiniteGeneric(_supp, shape=_supp.shape[self._p.ndim:])

        self._supp_flat = self.supp.reshape((self._p.size, self.size))
        if len(self._supp_flat) != len(np.unique(self._supp_flat, axis=0)):
            raise ValueError("Input 'supp' must have unique values")

        self._update_attr()

    # Input properties
    @property
    def supp(self):
        return self.space.support

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

        self._mode = self._supp_flat[np.argmax(self._p_flat)].reshape(self.shape)

    def _rvs(self, n, rng):
        return rng.choice(self._supp_flat, size=n, p=self._p_flat)

    # def pf(self, x, check_set=True):
    #     x, set_shape = check_data_shape(x, self.shape)
    #     x = x.reshape(-1, self.size)
    #
    #     out = np.empty(len(x))
    #     for i, x_i in enumerate(x):
    #         eq_supp = np.all(x_i == self._supp_flat, axis=-1)
    #         if check_set and eq_supp.sum() != 1:
    #             raise ValueError("Input 'x' must be in the support.")
    #
    #         out[i] = self._p_flat[eq_supp].squeeze()
    #
    #     return out.reshape(set_shape)

    def _pf_single(self, x):
        eq_supp = np.all(x.flatten() == self._supp_flat, axis=-1)
        if eq_supp.sum() == 0:
            raise ValueError("Input 'x' must be in the support.")

        return self._p_flat[eq_supp].squeeze()


class FiniteRV(MixinRV, Finite):
    """
    Generic RV drawn from a finite support set using an explicitly defined PMF.
    """

    def _update_attr(self):
        super()._update_attr()

        mean_flat = (self._p_flat[:, np.newaxis] * self._supp_flat).sum(axis=0)
        self._mean = mean_flat.reshape(self.shape)

        ctr_flat = self._supp_flat - mean_flat
        outer_flat = (ctr_flat[:, np.newaxis] * ctr_flat[..., np.newaxis]).reshape(self._p.size, -1)
        self._cov = (self._p_flat[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self.shape)


# s = np.random.random((3, 1, 2))
# pp = np.random.random((3,))
# pp = pp / pp.sum()
# f = Finite(s, pp)
# f.rvs((2, 3))
# f.pf(f.rvs((4, 5)))
#
# s = plot.mesh_grid([0, 1], [0, 1, 2])
# p = np.random.random((2, 3))
# p = p / p.sum()
# # s, p = ['a','b','c'], [.3,.2,.5]
# f2 = Finite(s, p)
# f2.pf(f2.rvs(4))
# f2.plot_pf()


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

        self._cov = (diag_gen(self._mean) - outer_gen(self._mean, self._mean)) / (self._alpha_0 + 1)

        self._log_pf_coef = gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))

    def _rvs(self, n, rng):
        return rng.dirichlet(self._alpha_0 * self._mean.flatten(), size=n)

    def pf(self, x):
        x, set_shape = _dirichlet_check_input(x, self._alpha_0, self._mean)

        log_pf = self._log_pf_coef + np.sum(xlogy(self._alpha_0 * self._mean - 1, x).reshape(-1, self.size), -1)
        return np.exp(log_pf).reshape(set_shape)

    def plot_pf(self, x=None, ax=None):
        if x is None:
            x = plot.simplex_grid(60, self.shape, hull_mask=(self.mean < 1 / self.alpha_0))
        return super().plot_pf(x, ax)


# rng = np.random.default_rng()
# a0 = 10
# m = np.random.random(3)
# m = m / m.sum()
# d = Dirichlet(a0, m, rng)
# d.plot_pf()
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pf(d.rvs())
# d.pf(d.rvs(4).reshape((2, 2)+d.mean.shape))


def _empirical_check_input(x, n, mean):
    x, set_shape = check_valid_pmf(x, data_shape=mean.shape)

    # if ((n * x) % 1 > 0).any():
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

        self._cov = (diag_gen(self._mean) - outer_gen(self._mean, self._mean)) / self._n

    def _rvs(self, n, rng):
        return rng.multinomial(self._n, self._mean.flatten(), size=n) / self._n

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

        self._cov = ((1/self._n + 1/self._alpha_0) / (1 + 1/self._alpha_0)
                     * (diag_gen(self._mean) - outer_gen(self._mean, self._mean)))

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
        log_pf = xlog1py(self._b - 1.0, -x) + xlogy(self._a - 1.0, x) - betaln(self._a, self._b)
        return np.exp(log_pf)


class Normal(BaseRV):
    def __init__(self, mean=0., cov=1., rng=None):
        """
        Normal random variable.

        Parameters
        ----------
        mean : float or np.ndarray
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

    def __str__(self):
        return f"NormalRV(mean={self.mean}, cov={self.cov})"

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

    def _rvs(self, n, rng):
        return rng.multivariate_normal(self._mean_flat, self._cov_flat, size=n)

    def pf(self, x):
        x, set_shape = check_data_shape(x, self.shape)

        dev = x.reshape(-1, self.size) - self._mean_flat
        maha = np.sum(np.square(np.dot(dev, self.prec_U)), axis=-1)

        log_pf = self._log_pf_coef + -0.5 * maha.reshape(set_shape)
        return np.exp(log_pf)

    def plot_pf(self, x=None, ax=None):
        if x is None and self.shape in ((), (2,)):
            if self.shape == ():
                lims = self._mean.item() + np.array([-1, 1]) * 3 * np.sqrt(self._cov.item())
            else:   # self.shape == (2,):
                lims = [(self._mean[i] - 3 * np.sqrt(self._cov[i, i]), self._mean[i] + 3 * np.sqrt(self._cov[i, i]))
                        for i in range(2)]

            x = plot.box_grid(lims, 100, endpoint=False)

        return super().plot_pf(x, ax)

# mean_, cov_ = 1., 1.
# # mean_, cov_ = np.ones(2), np.eye(2)
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


class NormalLinear(Normal):
    def __init__(self, weights=(0.,), basis=np.ones(1), cov=(1.,), rng=None):
        # self._set_weights(weights)
        # self._set_basis(basis)
        self._basis = np.array(basis)

        _mean_temp = np.empty(self._basis.shape[:-1])
        super().__init__(_mean_temp, cov, rng)

        self.weights = weights

    def __str__(self):
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

# class EmpiricalFinite(Finite):
#     def __init__(self, d, rng=None):
#         self.n = len(d)
#         self.values, self.counts = self._count_data(d)
#
#         super().__init__(self.values, self.counts / self.n)
#
#     @staticmethod
#     def _count_data(d):
#         return np.unique(d, return_counts=True, axis=0)


class GenericEmpirical(Base):       # TODO: implement using factory of Finite object?

    def __new__(cls, d, space=None, rng=None):
        if np.issubdtype(np.array(d).dtype, np.number):
            return super().__new__(GenericEmpiricalRV)
        else:
            return super().__new__(cls)

    def __init__(self, d, space=None, rng=None):
        super().__init__(rng)

        if space is None:
            self._space = spaces.Euclidean(d[0].shape)       # TODO: subclass for FiniteGeneric space?
        else:
            self._space = space

        self.n = len(d)
        self.values, self.counts = self._count_data(d)

        self._values_flat = self.values.reshape(-1, self.size)
        self.p = self.counts / self.n

        self._update_stats()

    def _update_stats(self):
        self._mode = self.values[self.counts.argmax()]

    @staticmethod
    def _count_data(d):
        return np.unique(d, return_counts=True, axis=0)

    def add_data(self, d):
        self.n += len(d)

        values_new, counts_new = [], []
        for value, count in zip(*self._count_data(d)):
            eq = np.all(value.flatten() == self._values_flat, axis=-1)
            if eq.sum() == 1:
                self.counts[eq] += count
            elif eq.sum() == 0:
                values_new.append(value)
                counts_new.append(count)
            else:
                raise ValueError

        if len(values_new) > 0:
            values_new, counts_new = np.array(values_new), np.array(counts_new)
            self.values = np.concatenate((self.values, values_new), axis=0)
            self.counts = np.concatenate((self.counts, counts_new), axis=0)

            self._values_flat = np.concatenate((self._values_flat, values_new.reshape(-1, self.size)), axis=0)
            self.p = self.counts / self.n

        self._update_stats()        # TODO: add efficient updates

    def _rvs(self, size, rng):
        return rng.choice(self.values, size, p=self.p)

    def pf(self, x):
        x, set_shape = check_data_shape(x, self.shape)
        x = x.reshape((math.prod(set_shape), *self.shape))

        out = np.zeros(len(x))
        for i, x_i in enumerate(x):
            if x_i in self.space:
                eq_supp = np.all(x_i.flatten() == self._values_flat, axis=-1)
                if eq_supp.sum() > 0:
                    out[i] = self.p[eq_supp].squeeze()
            else:
                raise ValueError("Must be in support.")

        return out.reshape(set_shape)

    # def _pf_single(self, x):
    #     eq_supp = np.all(x.flatten() == self._values_flat, axis=-1)
    #     if eq_supp.sum() != 1:
    #         raise ValueError("Input 'x' must be in the support.")
    #
    #     return self.p[eq_supp].squeeze()


class GenericEmpiricalRV(MixinRV, GenericEmpirical):
    def _update_stats(self):
        super()._update_stats()

        mean_flat = (self._values_flat * self.p[:, np.newaxis]).sum(0)
        self._mean = mean_flat.reshape(self.shape)

        ctr_flat = self._values_flat - mean_flat
        outer_flat = (ctr_flat[:, np.newaxis] * ctr_flat[..., np.newaxis]).reshape(len(ctr_flat), -1)
        self._cov = (self.p[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self.shape)


# # s = np.random.default_rng().random((2, 1, 3))
# # s = [[0, 1], [3, 6]]
# s = [0, .5]
# # s = ['a', 'b']
# rng_ = np.random.default_rng()
# d_ = rng_.choice(s, size=10, p=[.5, .5])
# e = GenericEmpirical(d_, space=spaces.FiniteGeneric([0,.1,.5,.6]))
# e.add_data(d_)
# print(e)
# e.plot_pf()


class Mixture(Base):
    def __new__(cls, dists, weights, rng=None):
        if all(isinstance(dist, BaseRV) for dist in dists):
            return super().__new__(MixtureRV)
        else:
            return super().__new__(cls)

    def __init__(self, dists, weights, rng=None):       # TODO: special implementation for Finite? get modes, etc?
        super().__init__(rng)
        self.dists = dists      # FIXME: updates on set??? `_set_dist` method or something?

        self._space = self.dists[0].space
        if not all(dist.space == self.space for dist in self.dists[1:]):
            raise ValueError("All distributions must have the same space.")

        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = np.array(value)
        if self._weights.sum() != 1:
            raise ValueError("Weights must sum to one.")

        if self._weights.shape == (len(self.dists),):
            self.n_dists = self.weights.size
        else:
            raise ValueError

    def _update_stats(self):
        self._mode = None  # TODO: formula???

    def pf(self, x):        # TODO: check plotting
        return sum(weight * dist.pf(x) for dist, weight in zip(self.dists, self._weights))

    def _rvs(self, n, rng):
        c = rng.choice(self.n_dists, size=n, p=self._weights)
        out = np.empty((n, *self.shape), dtype=self.space.dtype)
        for i, dist in enumerate(self.dists):
            idx = np.flatnonzero(c == i)
            out[idx] = dist.rvs(size=idx.size)

        return out


class MixtureRV(MixinRV, Mixture):
    def _update_stats(self):
        super()._update_stats()

        self._mean = sum(weight * dist.mean for dist, weight in zip(self.dists, self._weights))
        self._cov = None  # TODO


# dists_ = [Normal(mean, 1) for mean in [0, 10]]
# # dists_ = [Finite(['a', 'b'], p=[p_, 1-p_]) for p_ in [.5, 1]]
# w = [.5, .5]
# m = Mixture(dists_, w)
# m.rvs(10)
# m.plot_pf(x=np.linspace(-10, 20, 1001))
# pass
