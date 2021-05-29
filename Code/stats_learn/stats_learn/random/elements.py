"""
Random elements.
"""

# TODO: do ABC or PyCharm bug?

import math
from numbers import Integral
from typing import Optional, Union

import numpy as np
from scipy.special import gammaln, xlogy, xlog1py, betaln
from scipy.stats._multivariate import _PSD

from stats_learn.util import plotting, spaces
from stats_learn.util.base import DELTA, RandomGeneratorMixin, check_data_shape, check_valid_pmf, vectorize_func


# %% Base RE classes

class Base(RandomGeneratorMixin):
    """
    Base class for random element objects.
    """

    def __init__(self, rng=None):
        super().__init__(rng)

        self._space = None  # TODO: arg?

        self._mode = None  # TODO: make getter do numerical approx if None!?

    space = property(lambda self: self._space)

    shape = property(lambda self: self._space.shape)
    size = property(lambda self: self._space.size)
    ndim = property(lambda self: self._space.ndim)

    dtype = property(lambda self: self._space.dtype)

    mode = property(lambda self: self._mode)

    def pf(self, x):  # TODO: perform input checks using `space.__contains__`?
        # if x is None:
        #     x = self.space.x_plt  # TODO: add default x_plt

        return vectorize_func(self._pf_single, self.shape)(x)  # TODO: decorator? better way?

    def _pf_single(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass

    def plot_pf(self, x=None, ax=None, **kwargs):
        return self.space.plot(self.pf, x, ax, **kwargs)

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
        return rvs.reshape(shape + rvs.shape[1:])  # TODO: use np.asscalar if possible?

    def _rvs(self, n, rng):
        raise NotImplementedError("Method must be overwritten.")
        pass


class MixinRV:
    _mean: Optional[Union[float, np.ndarray]]
    _cov: Optional[Union[float, np.ndarray]]

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


# %% Specific RE's

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
class Finite(Base):  # TODO: DRY - use stat approx from the Finite space's methods?
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

    def __eq__(self, other):
        if isinstance(other, Finite):
            return np.all(self.supp == other.supp) and np.all(self.p == other.p)
        return NotImplemented

    def __deepcopy__(self, memodict=None):
        # if memodict is None:
        #     memodict = {}
        return type(self)(self.supp, self.p, self.rng)

    def __repr__(self):
        return f"FiniteRE(support={self.supp}, p={self.p})"

    # Input properties
    @property
    def supp(self):
        return self.space.values

    @property
    def _supp_flat(self):
        return self.space.values_flat

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

        self._mode = None
        # self._mode = self._supp_flat[np.argmax(self._p_flat)]

    @property
    def mode(self):
        if self._mode is None:
            self._mode = self._supp_flat[np.argmax(self._p_flat)]

        return self._mode

    def _rvs(self, n, rng):
        return rng.choice(self._supp_flat, size=n, p=self._p_flat)

    def _pf_single(self, x):
        eq_supp = np.all(x == self._supp_flat, axis=tuple(range(1, 1 + self.ndim)))
        # eq_supp = np.empty(self.space.set_size, dtype=np.bool)
        # for i, val in enumerate(self._supp_flat):
        #     eq_supp[i] = np.allclose(x, val)

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

        # self._mean = np.tensordot(self._p_flat, self._supp_flat, axes=[0, 0])
        #
        # ctr = self._supp_flat - self._mean
        # # outer = np.empty((len(ctr), *(2 * self.shape)))
        # # for i, ctr_i in enumerate(ctr):
        # #     outer[i] = np.tensordot(ctr_i, ctr_i, 0)
        # # self._cov = np.tensordot(self._p, outer, axes=[0, 0])
        # self._cov = sum(p_i * np.tensordot(ctr_i, ctr_i, 0) for p_i, ctr_i in zip(self._p_flat, ctr))

        self._mean = None
        self._cov = None

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.tensordot(self._p_flat, self._supp_flat, axes=(0, 0))

        return self._mean

    @property
    def cov(self):
        if self._cov is None:
            ctr = self._supp_flat - self.mean
            self._cov = sum(p_i * np.tensordot(ctr_i, ctr_i, 0) for p_i, ctr_i in zip(self._p_flat, ctr))

        return self._cov


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


def _dirichlet_check_input(x, mean, alpha_0):
    x, set_shape = check_valid_pmf(x, shape=mean.shape)

    if np.logical_and(x == 0, mean < 1 / alpha_0).any():
        raise ValueError("Each element in 'x' must be greater than "
                         "zero if the corresponding mean element is less than 1 / alpha_0.")

    return x, set_shape


class Dirichlet(BaseRV):
    """
    Dirichlet random process, finite-supp realizations.
    """

    def __init__(self, mean, alpha_0, rng=None):
        super().__init__(rng)
        self._space = spaces.Simplex(np.array(mean).shape)

        self._alpha_0 = alpha_0
        self._mean = check_valid_pmf(mean, full_support=True)
        self._update_attr()

    def __repr__(self):
        return f"Dirichlet(mean={self.mean}, alpha_0={self.alpha_0})"

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
            self._mode = None  # TODO: complete with general formula

        self._cov = (np.diagflat(self._mean).reshape(2 * self.shape)
                     - np.tensordot(self._mean, self._mean, 0)) / (self._alpha_0 + 1)

        self._log_pf_coef = gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))

        self.space.x_plt = None

    def _rvs(self, n, rng):
        return rng.dirichlet(self._alpha_0 * self._mean.flatten(), size=n).reshape(n, *self.shape)

    def pf(self, x):
        x, set_shape = _dirichlet_check_input(x, self._mean, self._alpha_0)

        log_pf = self._log_pf_coef + np.sum(xlogy(self._alpha_0 * self._mean - 1, x).reshape(-1, self.size), -1)
        return np.exp(log_pf).reshape(set_shape)

    def plot_pf(self, x=None, ax=None, **kwargs):
        if x is None and self.space._x_plt is None:
            self.space.x_plt = plotting.simplex_grid(self.space.n_plot, self.shape,
                                                     hull_mask=(self.mean < 1 / self.alpha_0))
        return self.space.plot(self.pf, x, ax)


# rng_ = np.random.default_rng()
# a0 = 10
# m = np.random.random(3)
# m = m / m.sum()
# d = Dirichlet(m, a0, rng_)
# d.plot_pf()
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pf(d.rvs())
# d.pf(d.rvs(4).reshape((2, 2)+d.mean.shape))


def _empirical_check_input(x, n, mean):
    x, set_shape = check_valid_pmf(x, shape=mean.shape)

    if (np.minimum((n * x) % 1, (-n * x) % 1) > 1e-9).any():
        raise ValueError("Each entry in 'x' must be a multiple of 1/n.")

    return x, set_shape


class Empirical(BaseRV):
    """
    Empirical random process, finite-supp realizations.
    """

    def __init__(self, mean, n, rng=None):
        super().__init__(rng)

        self._n = n
        self._mean = check_valid_pmf(mean)

        self._space = spaces.SimplexDiscrete(self.n, self.mean.shape)

        self._update_attr()

    def __repr__(self):
        return f"Empirical(mean={self.mean}, n={self.n})"

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
# d = Empirical(m, n, rng)
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

    def __init__(self, mean, alpha_0, n, rng=None):
        super().__init__(rng)
        self._space = spaces.SimplexDiscrete(n, np.array(mean).shape)

        self._mean = check_valid_pmf(mean)
        self._alpha_0 = alpha_0
        self._n = n
        self._update_attr()

    def __repr__(self):
        return f"DirichletEmpirical(mean={self.mean}, alpha_0={self.alpha_0}, n={self.n})"

    # Input properties
    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = check_valid_pmf(mean)
        if self._mean.shape != self.shape:
            raise ValueError(f"Mean shape must be {self.shape}.")
        self._update_attr()

    @property
    def alpha_0(self):
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = alpha_0
        self._update_attr()

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        # TODO: mode?

        self._cov = ((self._n + self._alpha_0) / self._n / (1 + self._alpha_0)
                     * (np.diagflat(self._mean).reshape(2 * self.shape) - np.tensordot(self._mean, self._mean, 0)))

        self._log_pf_coef = (gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))
                             + gammaln(self._n + 1) - gammaln(self._alpha_0 + self._n))

    def _rvs(self, n, rng):
        theta_flat = rng.dirichlet(self._alpha_0 * self._mean.flatten())
        return rng.multinomial(self._n, theta_flat, size=n).reshape(n, *self.shape) / self._n

    def pf(self, x):
        x, set_shape = _empirical_check_input(x, self._n, self._mean)

        log_pf = self._log_pf_coef + (gammaln(self._alpha_0 * self._mean + self._n * x)
                                      - gammaln(self._n * x + 1)).reshape(-1, self.size).sum(axis=-1)
        return np.exp(log_pf).reshape(set_shape)


# rng_ = np.random.default_rng()
# n = 10
# a0 = 600
# m = np.ones((3,))
# m = m / m.sum()
# d = DirichletEmpirical(m, a0, n, rng_)
# d.plot_pf()
# d.mean
# d.mode
# d.cov


class DirichletEmpiricalScalar(BaseRV):
    """
    Scalar Dirichlet-Empirical random variable.
    """

    def __init__(self, mean, alpha_0, n, rng=None):
        super().__init__(rng)

        self._multi = DirichletEmpirical([mean, 1 - mean], alpha_0, n, rng)
        self._space = spaces.FiniteGeneric(np.arange(n + 1) / n)

    def __repr__(self):
        return f"DirichletEmpiricalScalar(mean={self.mean}, alpha_0={self.alpha_0}, n={self.n})"

    # Input properties
    @property
    def mean(self):
        return self._multi.mean[0]

    @mean.setter
    def mean(self, mean):
        self._multi.mean = [mean, 1 - mean]

    @property
    def alpha_0(self):
        return self._multi.alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._multi.alpha_0 = alpha_0

    @property
    def n(self):
        return self._multi.n

    @n.setter
    def n(self, n):
        self._multi.n = n

    # Attribute Updates
    @property
    def cov(self):
        return self._multi.cov[0, 0]

    def _rvs(self, n, rng):
        a, b = self.alpha_0 * self._multi.mean
        p = rng.beta(a, b)
        return rng.binomial(self.n, p, size=n) / self.n

    def pf(self, x):
        x = np.array(x)
        return self._multi.pf(np.stack((x, 1 - x), axis=-1))


# de = DirichletEmpiricalScalar(.8, 5, 10)
# de.pf(.3)
# de.plot_pf()


class Beta(BaseRV):
    """
    Beta random variable.
    """

    def __init__(self, a=1, b=1, rng=None):
        super().__init__(rng)
        self._space = spaces.Box((0, 1))

        if a <= 0 or b <= 0:
            raise ValueError("Parameters must be strictly positive.")
        self._a = a
        self._b = b

        self._update_attr()

    def __repr__(self):
        return f"Beta({self.a}, {self.b})"

    @classmethod
    def from_mean(cls, mean, alpha_0, rng=None):
        return cls(alpha_0 * mean, alpha_0 * (1 - mean), rng)

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
        # self._alpha_0 = self._a + self._b

        if self._a > 1:
            if self._b > 1:
                self._mode = (self._a - 1) / (self._a + self._b - 2)
            else:
                self._mode = 1
        elif self._a <= 1:
            if self._b > 1:
                self._mode = 0
            elif self._a == 1 and self._b == 1:
                self._mode = 0  # any in unit interval
            else:
                self._mode = 0  # any in {0,1}

        self._mean = self._a / (self._a + self._b)
        self._cov = self._a * self._b / (self._a + self._b) ** 2 / (self._a + self._b + 1)

    def _rvs(self, n, rng):
        return rng.beta(self._a, self._b, size=n)

    def pf(self, x):
        x = np.array(x)
        log_pf = xlog1py(self._b - 1.0, -x) + xlogy(self._a - 1.0, x) - betaln(self._a, self._b)
        return np.exp(log_pf)


class Binomial(BaseRV):
    """
    Binomial random variable.
    """

    def __init__(self, p, n, rng=None):
        super().__init__(rng)
        self._space = spaces.FiniteGeneric(np.arange(n + 1))

        if n < 0:
            raise ValueError
        elif p < 0 or p > 1:
            raise ValueError
        self._n = n
        self._p = p

        self._update_attr()

    def __repr__(self):
        return f"Binomial(p={self.p}, n={self.n})"

    # Input properties
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        if n < 0:
            raise ValueError
        self._n = n
        self._update_attr()

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        if p < 0 or p > 1:
            raise ValueError
        self._p = p
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        _val = (self._n + 1) * self._p
        if _val == 0 or _val % 1 != 0:
            self._mode = math.floor(_val)
        elif _val - 1 in range(self._n):
            self._mode = _val
        elif _val - 1 == self._n:
            self._mode = self._n

        self._mean = self._n * self._p
        self._cov = self._n * self._p * (1 - self._p)

    def _rvs(self, n, rng):
        return rng.binomial(self._n, self._p, size=n)

    def pf(self, x):
        x = np.floor(x)
        combiln = (gammaln(self._n + 1) - (gammaln(x + 1) + gammaln(self._n - x + 1)))
        log_pf = combiln + xlogy(x, self._p) + xlog1py(self._n - x, -self._p)
        return np.exp(log_pf)


class EmpiricalScalar(Binomial):
    """
    Scalar empirical random variable. Equivalent to normalized Binomial RV.
    """

    def __init__(self, mean, n, rng=None):
        super().__init__(mean, n, rng)
        if self.n == 0:
            raise ValueError
        self._space = spaces.FiniteGeneric(np.arange(n + 1) / n)

    def __repr__(self):
        return f"EmpiricalScalar(mean={self.p}, n={self.n})"

    def __eq__(self, other):
        if isinstance(other, EmpiricalScalar):
            return self.n == other.n and self.mean == other.mean
        return NotImplemented

    def _update_attr(self):
        super()._update_attr()
        self._mode /= self._n
        self._mean /= self._n
        self._cov /= (self._n ** 2)

    def _rvs(self, n, rng):
        return super()._rvs(n, rng) / self._n

    def pf(self, x):
        x = np.array(x) * self._n
        return super().pf(x)


class Uniform(BaseRV):
    """
    Uniform random variable.
    """

    def __init__(self, lims, rng=None):
        super().__init__(rng)
        self._space = spaces.Box(lims)
        self._update_attr()

    def __repr__(self):
        return f"Uniform({self.lims})"

    # Input properties
    @property
    def lims(self):
        return self.space.lims

    @lims.setter
    def lims(self, val):
        self.space.lims = val
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        self._mean = np.mean(self.lims, axis=-1)
        self._mode = self._mean
        _temp = (self.lims[..., 1] - self.lims[..., 0]).flatten() ** 2 / 12
        self._cov = np.diag(_temp).reshape(2 * self.shape)

    def _rvs(self, n, rng):
        a_flat = self.lims[..., 0].flatten()
        b_flat = self.lims[..., 1].flatten()
        _temp = np.stack(tuple(rng.uniform(a, b, size=n) for a, b in zip(a_flat, b_flat)), axis=-1)
        return _temp.reshape((n, *self.shape))
        # return rng.uniform(self._a, self._b, size=n)

    def pf(self, x):
        val = 1 / np.prod(self.lims[..., 1] - self.lims[..., 0])

        x, set_shape = check_data_shape(x, self.shape)
        if not np.all((x >= self.lims[..., 0])) and np.all((x <= self.lims[..., 1])):
            raise ValueError(f"Values must be in interval {self.lims}")

        return np.full(set_shape, val)


class Normal(BaseRV):
    def __init__(self, mean=0., cov=1., *, allow_singular=False, rng=None):
        """
        Normal random variable.

        Parameters
        ----------
        *
        allow_singular
        mean : float or Iterable of float
            Mean
        cov : float or np.ndarray
            Covariance
        rng : np.random.Generator or int, optional
            Random number generator

        """

        super().__init__(rng)
        self.allow_singular = allow_singular

        self._space = spaces.Euclidean(np.array(mean).shape)

        self.mean = mean
        # self._mean = np.array(mean)
        # self._mean_flat = self._mean.flatten()

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

        if hasattr(self, '_cov'):
            self._set_lims_plot()  # avoids call before cov is set

    @property
    def cov(self):
        return self._cov

    @cov.setter
    # @BaseRV.cov.setter
    def cov(self, cov):
        self._cov = np.array(cov)

        if self._cov.shape == () and self.ndim == 1:  # TODO: hack-ish?
            self._cov = self._cov * np.eye(self.size)

        if self._cov.shape != self.shape * 2:
            raise ValueError(f"Covariance array shape must be {self.shape * 2}.")
        self._cov_flat = self._cov.reshape(2 * (self.size,))
        # self._cov_flat = self._cov.reshape(2 * self.shape)

        psd = _PSD(self._cov_flat, allow_singular=self.allow_singular)
        self.prec_U = psd.U
        self._log_pf_coef = -0.5 * (psd.rank * np.log(2 * np.pi) + psd.log_pdet)

        self._set_lims_plot()

    def _rvs(self, n, rng):
        return rng.multivariate_normal(self._mean_flat, self._cov_flat, size=n).reshape(n, *self.shape)

    def pf(self, x):
        x, set_shape = check_data_shape(x, self.shape)

        dev = x.reshape(-1, self.size) - self._mean_flat
        maha = np.sum(np.square(np.dot(dev, self.prec_U)), axis=-1)

        log_pf = self._log_pf_coef + -0.5 * maha.reshape(set_shape)
        return np.exp(log_pf)

    def _set_lims_plot(self):
        if self.shape in {(), (2,)}:
            if self.shape == ():
                lims = self._mean.item() + np.array([-1, 1]) * 3 * np.sqrt(self._cov.item())
            else:  # self.shape == (2,):
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


class NormalLinear(Normal):  # TODO: rework, only allow weights and cov to be set?
    def __init__(self, weights=(0.,), basis=np.ones(1), cov=(1.,), rng=None):
        self._basis = np.array(basis)

        _mean_temp = np.empty(self._basis.shape[:-1])
        super().__init__(_mean_temp, cov, rng=rng)

        self.weights = weights

    def __repr__(self):
        return f"NormalLinear(weights={self.weights}, basis={self.basis}, cov={self.cov})"

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, val):
        self._weights = np.array(val)
        if self._weights.ndim != 1:
            raise ValueError("Weights must be 1-dimensional.")
        self.mean = self._basis @ self._weights

    @property
    def basis(self):
        return self._basis


# bs = [[1, 0], [0, 1], [1, 1]]
# a = NormalLinear(weights=np.ones(2), basis=np.array(bs), cov=np.eye(3))


class DataEmpirical(Base):

    # TODO: subclass for FiniteGeneric space?

    def __new__(cls, values, counts, space=None, rng=None):
        if space is not None:
            dtype = space.dtype
        else:
            dtype = np.array(values).dtype

        if np.issubdtype(dtype, np.number):
            return super().__new__(DataEmpiricalRV)
        else:
            return super().__new__(cls)

    def __init__(self, values, counts, space=None, rng=None):
        super().__init__(rng)

        values, counts = map(np.array, (values, counts))
        if space is None:
            if np.issubdtype(values.dtype, np.number):
                self._space = spaces.Euclidean(values.shape[1:])
            else:
                raise NotImplementedError
        else:
            self._space = space

        self._init_attr()

        self.n = 0
        self.data = self._structure_data([], [])
        self.add_values(values, counts)

    def __repr__(self):
        return f"DataEmpirical(space={self.space}, n={self.n})"

    @classmethod
    def from_data(cls, d, space=None, rng=None):
        return cls(*cls._count_data(d), space, rng)

    def _init_attr(self):
        self._mode = np.full(self.shape, np.nan)

    @staticmethod
    def _count_data(d):
        return np.unique(d, return_counts=True, axis=0)

    def _structure_data(self, values, counts):
        return np.array(list(zip(values, counts)), dtype=[('x', self.dtype, self.shape), ('n', np.int32,)])

    def _get_idx(self, value):
        idx = np.flatnonzero(np.all(value == self.data['x'], axis=tuple(range(1, 1 + self.ndim))))
        if idx.size == 1:
            return idx.item()
        elif idx.size == 0:
            return None
        else:
            raise ValueError

    def add_data(self, d):
        self.add_values(*self._count_data(d))

    def add_values(self, values, counts):
        values, counts = map(np.array, (values, counts))
        n_new = counts.sum(dtype=np.int32)
        if n_new == 0:
            return

        self.n += n_new

        # Increment existing value counts, flag new values
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

    def _update_attr(self):
        self._p = self.data['n'] / self.n

        self._mode = None
        # self._mode = self.data['x'][self.data['n'].argmax()]

        self.space.x_plt = None

    @property
    def mode(self):
        if self._mode is None:
            self._mode = self.data['x'][self.data['n'].argmax()]

        return self._mode

    def _rvs(self, size, rng):
        return rng.choice(self.data['x'], size, p=self._p)

    def _pf_single(self, x):
        idx = self._get_idx(x)
        if idx is not None:
            # return self._p[idx]

            # FIXME: delta use needs to account for space dimensionality!?
            # delta = DELTA if isinstance(self.space, spaces.Continuous) else 1.
            if isinstance(self.space, spaces.Continuous):
                raise NotImplementedError
            else:
                delta = 1.

            return self._p[idx] * delta
        else:
            return 0.

    def plot_pf(self, x=None, ax=None, **kwargs):
        if x is None and self.space.x_plt is None:
            # self.space.set_x_plot()
            if isinstance(self.space, spaces.Continuous) and self.shape in {()}:
                # add empirical values to the plot support (so impulses are not missed)
                self.space.x_plt = np.sort(np.unique(np.concatenate((self.space.x_plt, self.data['x']))))

        return self.space.plot(self.pf, x, ax)


class DataEmpiricalRV(MixinRV, DataEmpirical):
    def __repr__(self):
        return f"DataEmpiricalRV(space={self.space}, n={self.n})"

    def _init_attr(self):
        super()._init_attr()

        self._mean = np.full(self.shape, np.nan)
        self._cov = np.full(2 * self.shape, np.nan)

    def _update_attr(self):
        super()._update_attr()

        self._mean = None
        self._cov = None

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.tensordot(self._p, self.data['x'], axes=(0, 0))

        return self._mean

    @property
    def cov(self):
        if self._cov is None:
            ctr = self.data['x'] - self.mean
            self._cov = sum(p_i * np.tensordot(ctr_i, ctr_i, 0) for p_i, ctr_i in zip(self._p, ctr))
            # TODO: try np.einsum?

        return self._cov


# # r = Beta(5, 5)
# # # r = Finite(plotting.mesh_grid([0, 1], [3, 4, 5]), np.ones((2, 3)) / 6)
# # # r = Finite(['a', 'b'], [.6, .4])
# # e = DataEmpirical.from_data(r.rvs(10), space=r.space)
# # e.add_data(r.rvs(10))
#
# # e = DataEmpirical(['a', 'b'], [5, 6], space=spaces.FiniteGeneric(['a', 'b']))
# e = DataEmpirical([], [], space=spaces.FiniteGeneric([0, 1]))
# # e.add_values(['b', 'c'], [4, 1])
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

    def __init__(self, dists, weights, rng=None):  # TODO: special implementation for Finite? get modes, etc?
        super().__init__(rng)
        self._dists = list(dists)

        self._space = spaces.check_spaces(self.dists)

        self.weights = weights

    def __repr__(self):
        _str = "; ".join([f"{prob}: {dist}" for prob, dist in zip(self._p, self.dists)])
        return f"Mixture({_str})"

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

    def set_dist_attr(self, idx, **dist_kwargs):
        dist = self._dists[idx]
        for key, val in dist_kwargs.items():
            setattr(dist, key, val)
        self._update_attr()

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

    # def add_dist(self, dist, weight):
    #     self._dists.append(dist)
    #     self.weights.append(weight)
    #     self._update_attr()

    # def del_dist(self, idx):
    #     del self._dists[idx]
    #     del self.weights[idx]
    #     self._update_attr()

    @property
    def _idx_nonzero(self):
        return np.flatnonzero(self._weights)

    def _update_attr(self):
        self.space.x_plt = None

        self._p = self._weights / self._weights.sum()

        self._mode = None

    @property
    def mode(self):  # TODO: implement similar functionality throughout for costly dependent attributes!!!
        if self._mode is None:
            if self._idx_nonzero.size == 1:
                self._mode = self._dists[self._idx_nonzero.item()].mode
            else:
                self._mode = self.space.argmax(self.pf)

        return self._mode

    def _rvs(self, n, rng):
        idx_rng = rng.choice(self.n_dists, size=n, p=self._p)
        out = np.empty((n, *self.shape), dtype=self.space.dtype)
        for i, dist in enumerate(self.dists):
            idx = np.flatnonzero(i == idx_rng)
            if idx.size > 0:
                out[idx] = dist.rvs(size=idx.size)

        return out

    def pf(self, x):
        # return sum(prob * dist.pf(x) for prob, dist in zip(self._p, self.dists) if prob > 0)
        return sum(self._p[i] * self.dists[i].pf(x) for i in self._idx_nonzero)

    def plot_pf(self, x=None, ax=None, **kwargs):
        if x is None and self.space.x_plt is None:
            # self.space.set_x_plot()

            # dists_nonzero = [dist for (w, dist) in zip(self.weights, self.dists) if w > 0]
            dists_nonzero = [self.dists[i] for i in self._idx_nonzero]

            # Update space plotting attributes
            if isinstance(self.space, spaces.Euclidean):
                temp = np.stack(list(dist.space.lims_plot for dist in dists_nonzero))
                self._space.lims_plot = [temp[:, 0].min(), temp[:, 1].max()]

            if isinstance(self.space, spaces.Continuous) and self.shape in {()}:
                x = self.space.x_plt
                for dist in dists_nonzero:
                    if isinstance(dist, DataEmpirical):
                        x = np.concatenate((x, dist.data['x']))

                self.space.x_plt = np.sort(np.unique(x))

        return self.space.plot(self.pf, x, ax)


class MixtureRV(MixinRV, Mixture):
    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"MixtureRV({_str})"

    def _update_attr(self):
        super()._update_attr()

        self._mean = None
        self._cov = None

    @property
    def mean(self):
        if self._mean is None:
            # self._mean = sum(prob * dist.mean for prob, dist in zip(self._p, self.dists) if prob > 0)
            self._mean = sum(self._p[i] * self.dists[i].mean for i in self._idx_nonzero)

        return self._mean

# # dists_ = [Beta(*args) for args in [[10, 5], [2, 12]]]
# # dists_ = [Normal(mean, 1) for mean in [0, 4]]
# # dists_ = [Normal(mean, 1) for mean in [[0, 0], [2, 3]]]
# # dists_ = [Finite(['a', 'b'], p=[p_, 1-p_]) for p_ in [0, 1]]
# # dists_ = [Finite([[0, 0], [0, 1]], p=[p_, 1-p_]) for p_ in [0, 1]]
#
# dists_ = [Normal(2)]
# dists_.append(DataEmpirical.from_data(dists_[0].rvs(0)))
#
# m = Mixture(dists_, [5, 0])
# m.rvs(10)
# m.plot_pf()
# print(m.space.integrate(m.pf))
# print(m.space.moment(m.pf))
# qq = None
