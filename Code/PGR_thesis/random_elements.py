"""
Random element objects.
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

from util.generic import check_rng, check_data_shape, check_valid_pmf, vectorize_func, vectorize_func_dec
from util.math import outer_gen, diag_gen, simplex_round
from util.plot import simplex_grid


#%% Base RE classes

class Base:
    """
    Base class for generic random element objects.
    """

    def __init__(self, rng=None):
        self.rng = rng

        self._shape = None
        self._mode = None

    shape = property(lambda self: self._shape)
    size = property(lambda self: math.prod(self._shape))
    ndim = property(lambda self: len(self._shape))

    mode = property(lambda self: self._mode)

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng):
        self._rng = check_rng(rng)

    def pf(self, x):
        return vectorize_func(self._pf_single, self._shape)(x)     # TODO: decorator? better way?

    def _pf_single(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass

    def plot_pf(self, ax=None):
        raise NotImplementedError
        pass

    def rvs(self, size=None, rng=None):
        if size is None:
            size = ()
        elif isinstance(size, (Integral, np.integer)):
            size = (size,)
        elif not isinstance(size, tuple):
            raise TypeError("Input 'size' must be int or tuple.")

        if rng is None:
            rng = self.rng
        else:
            rng = check_rng(rng)

        return self._rvs(size, rng)

    def _rvs(self, size, rng):
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
        self._mean = None
        self._cov = None


#%% Specific RE's

class DeterministicRE(Base):
    """
    Deterministic random element.
    """

    # TODO: redundant, just use FiniteRE? or change to ContinuousRV for integration? General dirac mix?

    def __new__(cls, val, rng=None):
        val = np.asarray(val)
        if np.issubdtype(val.dtype, np.number):
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
        self._shape = self._val.shape
        self._mode = self._val

    def _rvs(self, size, rng):
        return np.broadcast_to(self._val, size + self._shape)

    def pf(self, x):
        return np.where(np.all(x.reshape(-1, self.size) == self._val.flatten(), axis=-1), 1., 0.)

    # def _pf_single(self, x):
    #     return 1. if np.all(np.array(x) == self._val) else 0.


class DeterministicRV(MixinRV, DeterministicRE):
    """
    Deterministic random variable.
    """

    @DeterministicRE.val.setter
    def val(self, val):
        DeterministicRE.val.fset(self, val)
        # super(DeterministicRV, self.__class__).val.fset(self, val)    # TODO: super instead?

        self._mean = self._val
        self._cov = np.zeros(2 * self._shape)


# rng = np.random.default_rng()
# a = np.arange(6).reshape(3, 2)
# # a = ['a','b','c']
# b = DeterministicRE(a[1], rng)
# b.mode
# b.mean
# b.cov
# b.pf(b.rvs(8))


class FiniteRE(Base):
    """
    Generic RE drawn from a finite support set using an explicitly defined PMF.
    """

    def __new__(cls, supp, p, rng=None):
        supp = np.array(supp)
        if np.issubdtype(supp.dtype, np.number):
            return super().__new__(FiniteRV)
        else:
            return super().__new__(cls)

    def __init__(self, supp, p, rng=None):
        super().__init__(rng)
        self._supp = np.array(supp)
        self._p = check_valid_pmf(p)
        self._update_attr()

    # Input properties
    @property
    def supp(self):
        return self._supp

    @supp.setter
    def supp(self, supp):
        self._supp = np.array(supp)
        self._update_attr()

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = check_valid_pmf(p)
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        set_shape, self._shape = self._supp.shape[:self._p.ndim], self._supp.shape[self._p.ndim:]

        if set_shape != self._p.shape:
            raise ValueError("Leading shape values of 'supp' must equal the shape of 'p'.")

        self._supp_flat = self._supp.reshape((self._p.size, -1))
        self._p_flat = self._p.flatten()
        if len(self._supp_flat) != len(np.unique(self._supp_flat, axis=0)):
            raise ValueError("Input 'supp' must have unique values")

        self._mode = self._supp_flat[np.argmax(self._p_flat)].reshape(self._shape)

    def _rvs(self, size, rng):
        i = rng.choice(self._p.size, size, p=self._p_flat)
        return self._supp_flat[i].reshape(size + self._shape)

    def _pf_single(self, x):
        eq_supp = np.all(x.flatten() == self._supp_flat, axis=-1)
        if eq_supp.sum() != 1:
            raise ValueError("Input 'x' must be in the support.")

        return self._p_flat[eq_supp].squeeze()

    def plot_pf(self, ax=None):
        if self._p.ndim == 1:
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x$', ylabel=r'$\mathrm{P}_\mathrm{x}(x)$')

            plt_data = ax.stem(self._supp, self._p, use_line_collection=True)

            return plt_data
        else:
            raise NotImplementedError('Plot method only implemented for 1-dimensional data.')


class FiniteRV(MixinRV, FiniteRE):
    """
    Generic RV drawn from a finite support set using an explicitly defined PMF.
    """

    def _update_attr(self):
        super()._update_attr()

        mean_flat = (self._p_flat[:, np.newaxis] * self._supp_flat).sum(axis=0)
        self._mean = mean_flat.reshape(self._shape)

        ctr_flat = self._supp_flat - mean_flat
        outer_flat = (ctr_flat[:, np.newaxis] * ctr_flat[..., np.newaxis]).reshape(self._p.size, -1)
        self._cov = (self._p_flat[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self._shape)

    def plot_pf(self, ax=None):
        if self._p.ndim == 1:
            super().plot_pf(ax)

        elif self._p.ndim == 2:
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel=r'$\mathrm{P}_\mathrm{x}(x)$')

            plt_data = ax.bar3d(self._supp[..., 0].flatten(),
                                self._supp[..., 1].flatten(), 0, 1, 1, self._p_flat, shade=True)
            return plt_data

        elif self._p.ndim == 3:
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

            plt_data = ax.scatter(self._supp[..., 0], self._supp[..., 1], self._supp[..., 2], s=15, c=self._p)

            c_bar = plt.colorbar(plt_data)
            c_bar.set_label(r'$\mathrm{p}_\mathrm{x}(x)$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')


# s = np.random.random((1, 1, 2))
# pp = np.random.random((1,))
# pp = pp / pp.sum()
# f = FiniteRE(s, pp)
# f.pf(f.rvs((4, 5)))
#
# s = np.stack(np.meshgrid([0, 1], [0, 1], [0, 1]), axis=-1)
# p = np.random.random((2, 2, 2))
# p = p / p.sum()
# # s, p = ['a','b','c'], [.3,.2,.5]
# f2 = FiniteRE(s, p)
# f2.pf(f2.rvs(4))
# f2.plot_pf()


def _dirichlet_check_alpha_0(alpha_0):
    alpha_0 = np.asarray(alpha_0)
    if alpha_0.size > 1 or alpha_0 <= 0:
        raise ValueError("Concentration parameter must be a positive scalar.")
    return alpha_0


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
        self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
        self._mean = check_valid_pmf(mean, full_support=True)
        self._update_attr()

    # Input properties
    @property
    def alpha_0(self):
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
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
        self._shape = self._mean.shape

        if np.min(self._mean) > 1 / self._alpha_0:
            self._mode = (self._mean - 1 / self._alpha_0) / (1 - self.size / self._alpha_0)
        else:
            # warnings.warn("Mode method currently supported for mean > 1/alpha_0 only")
            self._mode = None       # TODO: complete with general formula

        self._cov = (diag_gen(self._mean) - outer_gen(self._mean, self._mean)) / (self._alpha_0 + 1)

        self._log_pf_coef = gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))

    def _rvs(self, size, rng):
        return rng.dirichlet(self._alpha_0 * self._mean.flatten(), size).reshape(size + self._shape)

    def pf(self, x):
        x, set_shape = _dirichlet_check_input(x, self._alpha_0, self._mean)

        log_pf = self._log_pf_coef + np.sum(xlogy(self._alpha_0 * self._mean - 1, x).reshape(-1, self.size), -1)
        return np.exp(log_pf).reshape(set_shape)

    def plot_pf(self, x=None, ax=None):
        n_plt = 40

        if self.size in (2, 3):
            if x is None:
                x = simplex_grid(n_plt, self._shape, hull_mask=(self.mean < 1 / self.alpha_0))

            pf_plt = self.pf(x)
            x.resize(x.shape[0], self.size)

            # pf_plt.sum() / (n_plt ** (self._size - 1))

            if self.size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x[:, 0], x[:, 1], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{p}_\mathrm{x}(x)$')

                return plt_data

            elif self.size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.view_init(35, 45)
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{p}_\mathrm{x}(x)$')

                return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')


# rng = np.random.default_rng()
# a0 = 10
# m = np.random.random((1, 3))
# m = m / m.sum()
# d = Dirichlet(a0, m, rng)
# d.plot_pf()
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pf(d.rvs())
# d.pf(d.rvs(4).reshape((2, 2)+d.mean.shape))


def _empirical_check_n(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError("Input 'n' must be a positive integer.")
    return n


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
        self._n = _empirical_check_n(n)
        self._mean = check_valid_pmf(mean)
        self._update_attr()

    # Input properties
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = _empirical_check_n(n)
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
        self._shape = self._mean.shape

        self._mode = ((self._n * self._mean) // 1) + simplex_round((self._n * self._mean) % 1)  # FIXME: broken

        self._cov = (diag_gen(self._mean) - outer_gen(self._mean, self._mean)) / self._n

        self._log_pf_coef = gammaln(self._n + 1)

    def _rvs(self, size, rng):
        return rng.multinomial(self._n, self._mean.flatten(), size).reshape(size + self._shape) / self._n

    def pf(self, x):
        x, set_shape = _empirical_check_input(x, self._n, self._mean)

        log_pf = self._log_pf_coef + (xlogy(self._n * x, self._mean)
                                      - gammaln(self._n * x + 1)).reshape(-1, self.size).sum(axis=-1)
        return np.exp(log_pf).reshape(set_shape)

    def plot_pf(self, ax=None):

        if self.size in (2, 3):
            x_plt = simplex_grid(self.n, self._shape)
            pf_plt = self.pf(x_plt)
            x_plt.resize(x_plt.shape[0], self.size)

            if self.size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

                return plt_data

            elif self.size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.view_init(35, 45)
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], x_plt[:, 2], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

                return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')

# rng = np.random.default_rng()
# n = 10
# # m = np.random.random((1, 3))
# m = np.random.default_rng().integers(10, size=(1, 3))
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
        self._n = _empirical_check_n(n)
        self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
        self._mean = check_valid_pmf(mean)
        self._update_attr()

    # Input properties
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = _empirical_check_n(n)
        self._update_attr()

    @property
    def alpha_0(self):
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
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
        self._shape = self._mean.shape

        # TODO: mode?

        self._cov = ((1/self._n + 1/self._alpha_0) / (1 + 1/self._alpha_0)
                     * (diag_gen(self._mean) - outer_gen(self._mean, self._mean)))

        self._log_pf_coef = (gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))
                             + gammaln(self._n + 1) - gammaln(self._alpha_0 + self._n))

    def _rvs(self, size, rng):
        # return rng.multinomial(self._n, self._mean.flatten(), size).reshape(size + self._shape) / self._n
        raise NotImplementedError

    def pf(self, x):
        x, set_shape = _empirical_check_input(x, self._n, self._mean)

        log_pf = self._log_pf_coef + (gammaln(self._alpha_0 * self._mean + self._n * x) - gammaln(self._n * x + 1))\
            .reshape(-1, self.size).sum(axis=-1)
        return np.exp(log_pf).reshape(set_shape)

    def plot_pf(self, ax=None):        # TODO: reused code. define simplex plotter outside!

        if self.size in (2, 3):
            x_plt = simplex_grid(self.n, self._shape)
            pf_plt = self.pf(x_plt)
            x_plt.resize(x_plt.shape[0], self.size)

            if self.size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

                return plt_data

            elif self.size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.view_init(35, 45)
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], x_plt[:, 2], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

                return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')

# rng_ = np.random.default_rng()
# n = 10
# a0 = 600
# m = np.ones((1, 3))
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
        if a <= 0 or b <= 0:
            raise ValueError("Parameters must be strictly positive.")
        self._a, self._b = a, b

        self._shape = ()
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

    def _rvs(self, size, rng):
        return rng.beta(self._a, self._b, size)

    def pf(self, x):
        log_pf = xlog1py(self._b - 1.0, -x) + xlogy(self._a - 1.0, x) - betaln(self._a, self._b)
        return np.exp(log_pf)

    def plot_pf(self, x=None, ax=None):
        if x is None:
            x = np.linspace(0, 1, 101, endpoint=True)
        if ax is None:
            _, ax = plt.subplots()
            ax.set(xlabel='$x$', ylabel=r'$\mathrm{P}_\mathrm{x}(x)$')

        plt_data = ax.plot(x, self.pf(x))
        return plt_data


class Normal(BaseRV):
    def __init__(self, mean=0., cov=1., rng=None):
        super().__init__(rng)
        self._shape = np.array(mean).shape

        self.mean = mean
        self.cov = cov

        # self._inv_cov = None
        # self._psd = None

    def __repr__(self):
        return f"NormalRV(mean={self.mean}, cov={self.cov})"

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = np.array(mean)
        if self._mean.shape != self._shape:
            raise ValueError(f"Mean array shape must be {self._shape}.")
        self._mean_flat = self._mean.flatten()

        self._mode = self._mean

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, cov):
        self._cov = np.array(cov)
        if self._cov.shape != self._shape * 2:
            raise ValueError(f"Covariance array shape must be {self._shape * 2}.")
        self._cov_flat = self._cov.reshape(2 * (self.size,))

        # self._psd = _PSD(self._cov)
        # self._log_pf_coef = -0.5 * (self._psd.rank * np.log(2 * np.pi) + self._psd.log_pdet)

        # self._inv_cov = inverse(self._cov)
        # _log_det_cov = np.log(determinant(self._cov))
        # self._inv_cov = np.linalg.inv(self._cov)
        # _log_det_cov = np.log(np.linalg.det(self._cov))
        # self._log_pf_coef = -0.5 * (self._size * np.log(2 * np.pi) + _log_det_cov)

        psd = _PSD(self._cov_flat, allow_singular=False)
        self.prec_U = psd.U
        self._log_pf_coef = -0.5 * (psd.rank * np.log(2 * np.pi) + psd.log_pdet)

    def _rvs(self, size, rng):
        return rng.multivariate_normal(self._mean_flat, self._cov_flat, size).reshape(size + self._shape)
        # if self._shape == ():
        #     return rng.normal(self._mean, np.sqrt(self._cov), size).reshape(size + self._shape)
        # else:
        #     return rng.multivariate_normal(self._mean, self._cov, size).reshape(size + self._shape)

    def pf(self, x):
        x, set_shape = check_data_shape(x, self._shape)

        dev = x.reshape(-1, self.size) - self._mean_flat
        maha = np.sum(np.square(np.dot(dev, self.prec_U)), axis=-1)

        # inner_white = np.array([(outer_gen(dev_i, np.ones(self._shape)) * self._inv_cov
        #                         * outer_gen(np.ones(self._shape), dev_i)).sum() for dev_i in dev])
        # inner_white = np.sum(np.square(np.dot(dev, self._psd.U)), axis=-1)

        log_pf = self._log_pf_coef + -0.5 * maha.reshape(set_shape)
        return np.exp(log_pf)

    @property
    def x_default(self):
        n_plt = 100

        if self.size == 1:
            lims = self._mean.item() + np.array([-1, 1]) * 3 * np.sqrt(self._cov.item())
            x = np.linspace(*lims, n_plt, endpoint=False).reshape(n_plt, *self._shape)
        elif self.size == 2:
            lims = [(self._mean[i] - 3 * np.sqrt(self._cov[i, i]), self._mean[i] + 3 * np.sqrt(self._cov[i, i]))
                    for i in range(2)]
            x0_plt = np.linspace(*lims[0], n_plt, endpoint=False)
            x1_plt = np.linspace(*lims[1], n_plt, endpoint=False)
            x = np.stack(np.meshgrid(x0_plt, x1_plt), axis=-1)
        else:
            raise NotImplementedError('Plot method only supported for 1- and 2-dimensional data.')

        return x

    def plot_pf(self, x=None, ax=None):
        # _delta = 0.01
        n_plt = 100

        if x is None:
            x = self.x_default

        if self.size == 1:
            # if x is None:
            #     lims = self._mean.item() + np.array([-1, 1]) * 3*np.sqrt(self._cov.item())
            #     # n_plt = int(round((lims[1]-lims[0]) / _delta))
            #     x = np.linspace(*lims, n_plt, endpoint=False).reshape(n_plt, *self._shape)

            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x_1$', ylabel='$p$')

            plt_data = ax.plot(x, self.pf(x))
            return plt_data

        elif self.size == 2:
            # if x is None:
            #     lims = [(self._mean[i] - 3 * np.sqrt(self._cov[i, i]), self._mean[i] + 3 * np.sqrt(self._cov[i, i]))
            #             for i in range(2)]
            #     # n_plt = int(round((lims[0][1] - lims[0][0]) / _delta)), int(round((lims[1][1] - lims[1][0]) / _delta))
            #     x0_plt = np.linspace(*lims[0], n_plt, endpoint=False)
            #     x1_plt = np.linspace(*lims[1], n_plt, endpoint=False)
            #     x = np.stack(np.meshgrid(x0_plt, x1_plt), axis=-1)

            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$p$')

            # ax.plot_wireframe(x[..., 0], x[..., 1], self.pf(x))
            plt_data = ax.plot_surface(x[..., 0], x[..., 1], self.pf(x), cmap=plt.cm.viridis)

            return plt_data
        else:
            raise NotImplementedError('Plot method only supported for 1- and 2-dimensional data.')


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
