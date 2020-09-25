"""
Random element objects.
"""

# TODO: docstrings?
# TODO: do ABC or PyCharm bug?

import numpy as np
from scipy.special import gammaln, xlogy, xlog1py, betaln
import matplotlib.pyplot as plt

from util.generic import check_rng, check_data_shape, check_valid_pmf, vectorize_func, vectorize_func_dec
from util.math import outer_gen, diag_gen, simplex_round, inverse, determinant
from util.plot import simplex_grid


#%% Base RE classes

class BaseRE:
    """
    Base class for generic random element objects.
    """

    def __init__(self, rng=None):
        self._rng = check_rng(rng)       # TODO: want scipy subclass for seed control functionality?

        self._data_shape = None
        self._data_size = None
        self._mode = None

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, rng):
        if rng is not None:
            self._rng = check_rng(rng)

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def mode(self):
        return self._mode

    def pf(self, x):
        return vectorize_func(self._pf_single, self._data_shape)(x)     # TODO: decorator? better way?

    def _pf_single(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass

    def plot_pf(self, ax=None):
        raise NotImplementedError
        pass

    def rvs(self, size=None, rng=None):
        if size is None:
            size = ()
        elif type(size) is int:
            size = (size,)
        elif type(size) is not tuple:
            raise TypeError("Input 'size' must be int or tuple.")

        self.rng = rng

        return self._rvs(size)

    def _rvs(self, size=None):
        raise NotImplementedError("Method must be overwritten.")
        pass


class MixinRV:
    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov


class BaseRV(MixinRV, BaseRE):
    """
    Base class for generic random variable (numeric) objects.
    """

    def __init__(self, rng=None):
        super().__init__(rng)
        self._mean = None
        self._cov = None

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov


#%% Specific RE's

class DeterministicRE(BaseRE):
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
        self._val = np.asarray(val)

        self._data_shape = self._val.shape
        self._data_size = self._val.size

        self._mode = self._val

    def _rvs(self, size=()):
        return np.broadcast_to(self._val, size + self._data_shape)

    # def pf(self, x):
    #     return np.where(np.all(x.reshape(-1, self._data_size) == self._val.flatten(), axis=-1), 1., 0.)

    def _pf_single(self, x):
        return 1. if np.all(np.array(x) == self._val) else 0.


class DeterministicRV(MixinRV, DeterministicRE):
    """
    Deterministic random variable.
    """

    @DeterministicRE.val.setter
    def val(self, val):
        DeterministicRE.val.fset(self, val)
        # super(DeterministicRV, self.__class__).val.fset(self, val)    # TODO: super instead?

        self._mean = self._val
        self._cov = np.zeros(2 * self._data_shape)


# rng = np.random.default_rng()
# a = np.arange(6).reshape(3, 2)
# # a = ['a','b','c']
# b = DeterministicRE(a[1], rng)
# b.mode
# b.mean
# b.cov
# b.pf(b.rvs(8))


class FiniteRE(BaseRE):
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
        set_shape, self._data_shape = self._supp.shape[:self._p.ndim], self._supp.shape[self._p.ndim:]
        self._data_size = int(np.prod(self._data_shape))

        if set_shape != self._p.shape:
            raise ValueError("Leading shape values of 'supp' must equal the shape of 'p'.")

        self._supp_flat = self._supp.reshape((self._p.size, -1))
        self._p_flat = self._p.flatten()
        if len(self._supp_flat) != len(np.unique(self._supp_flat, axis=0)):
            raise ValueError("Input 'supp' must have unique values")

        self._mode = self._supp_flat[np.argmax(self._p_flat)].reshape(self._data_shape)

    def _rvs(self, size=()):
        i = self.rng.choice(self._p.size, size, p=self._p_flat)
        return self._supp_flat[i].reshape(size + self._data_shape)

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
        self._mean = mean_flat.reshape(self._data_shape)

        ctr_flat = self._supp_flat - mean_flat
        outer_flat = (ctr_flat[:, np.newaxis] * ctr_flat[..., np.newaxis]).reshape(self._p.size, -1)
        self._cov = (self._p_flat[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self._data_shape)

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


class DirichletRV(BaseRV):
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
        self._data_shape = self._mean.shape
        self._data_size = self._mean.size

        if np.min(self._mean) > 1 / self._alpha_0:
            self._mode = (self._mean - 1 / self._alpha_0) / (1 - self._data_size / self._alpha_0)
        else:
            # warnings.warn("Mode method currently supported for mean > 1/alpha_0 only")
            self._mode = None       # TODO: complete with general formula

        self._cov = (diag_gen(self._mean) - outer_gen(self._mean, self._mean)) / (self._alpha_0 + 1)

        self._log_pf_coef = gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))

    def _rvs(self, size=()):
        return self.rng.dirichlet(self._alpha_0 * self._mean.flatten(), size).reshape(size + self._data_shape)

    def pf(self, x):
        x, set_shape = _dirichlet_check_input(x, self._alpha_0, self._mean)

        log_pf = self._log_pf_coef + np.sum(xlogy(self._alpha_0 * self._mean - 1, x).reshape(-1, self._data_size), -1)
        return np.exp(log_pf).reshape(set_shape)

    def plot_pf(self, x_plt=None, ax=None):

        if self._data_size in (2, 3):
            if x_plt is None:
                x_plt = simplex_grid(40, self._data_shape, hull_mask=(self.mean < 1 / self.alpha_0))
            # x_plt = simplex_grid(n_plt, self._data_shape, hull_mask=(self.mean < 1 / self.alpha_0))

            pf_plt = self.pf(x_plt)
            x_plt.resize(x_plt.shape[0], self._data_size)

            # pf_plt.sum() / (n_plt ** (self._data_size - 1))

            if self._data_size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{p}_\mathrm{x}(x)$')

                return plt_data

            elif self._data_size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.view_init(35, 45)
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], x_plt[:, 2], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{p}_\mathrm{x}(x)$')

                return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')


# rng = np.random.default_rng()
# a0 = 10
# m = np.random.random((1, 3))
# m = m / m.sum()
# d = DirichletRV(a0, m, rng)
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


class EmpiricalRV(BaseRV):
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
        self._data_shape = self._mean.shape
        self._data_size = self._mean.size

        self._mode = ((self._n * self._mean) // 1) + simplex_round((self._n * self._mean) % 1)  # FIXME: broken

        self._cov = (diag_gen(self._mean) - outer_gen(self._mean, self._mean)) / self._n

        self._log_pf_coef = gammaln(self._n + 1)

    def _rvs(self, size=()):
        return self.rng.multinomial(self._n, self._mean.flatten(), size).reshape(size + self._data_shape) / self._n

    def pf(self, x):
        x, set_shape = _empirical_check_input(x, self._n, self._mean)

        log_pf = self._log_pf_coef + (xlogy(self._n * x, self._mean)
                                      - gammaln(self._n * x + 1)).reshape(-1, self._data_size).sum(axis=-1)
        return np.exp(log_pf).reshape(set_shape)

    def plot_pf(self, ax=None):

        if self._data_size in (2, 3):
            x_plt = simplex_grid(self.n, self._data_shape)
            pf_plt = self.pf(x_plt)
            x_plt.resize(x_plt.shape[0], self._data_size)

            if self._data_size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

                return plt_data

            elif self._data_size == 3:
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
# d = EmpiricalRV(n, m, rng)
# d.plot_pf()
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pf(d.rvs())
# d.pf(d.rvs(4).reshape((2, 2) + d.mean.shape))


class DirichletEmpiricalRV(BaseRV):
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
        self._data_shape = self._mean.shape
        self._data_size = self._mean.size

        # TODO: mode?

        self._cov = ((1/self._n + 1/self._alpha_0) / (1 + 1/self._alpha_0)
                     * (diag_gen(self._mean) - outer_gen(self._mean, self._mean)))

        self._log_pf_coef = (gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))
                             + gammaln(self._n + 1) - gammaln(self._alpha_0 + self._n))

    def _rvs(self, size=()):
        # return rng.multinomial(self._n, self._mean.flatten(), size).reshape(size + self._data_shape) / self._n
        raise NotImplementedError

    def pf(self, x):
        x, set_shape = _empirical_check_input(x, self._n, self._mean)

        log_pf = self._log_pf_coef + (gammaln(self._alpha_0 * self._mean + self._n * x) - gammaln(self._n * x + 1))\
            .reshape(-1, self._data_size).sum(axis=-1)
        return np.exp(log_pf).reshape(set_shape)

    def plot_pf(self, ax=None):        # TODO: reused code. define simplex plotter outside!

        if self._data_size in (2, 3):
            x_plt = simplex_grid(self.n, self._data_shape)
            pf_plt = self.pf(x_plt)
            x_plt.resize(x_plt.shape[0], self._data_size)

            if self._data_size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

                return plt_data

            elif self._data_size == 3:
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
# d = DirichletEmpiricalRV(n, a0, m, rng_)
# d.plot_pf()
# d.mean
# d.mode
# d.cov


class BetaRV(BaseRV):
    """
    Beta random variable.
    """

    def __init__(self, a, b, rng=None):
        super().__init__(rng)
        if a <= 0 or b <= 0:
            raise ValueError("Parameters must be strictly positive.")
        self._a, self._b = a, b

        self._data_shape = ()
        self._data_size = 1
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

    def _rvs(self, size=()):
        return self.rng.beta(self._a, self._b, size)

    def pf(self, x):
        log_pf = xlog1py(self._b - 1.0, -x) + xlogy(self._a - 1.0, x) - betaln(self._a, self._b)
        return np.exp(log_pf)

    def plot_pf(self, x_plt=None, ax=None):
        if x_plt is None:
            x_plt = np.linspace(0, 1, 101, endpoint=True)
        if ax is None:
            _, ax = plt.subplots()
            ax.set(xlabel='$x$', ylabel=r'$\mathrm{P}_\mathrm{x}(x)$')

        plt_data = ax.plot(x_plt, self.pf(x_plt))
        return plt_data


class NormalRV(BaseRV):
    def __init__(self, mean=0., cov=1., rng=None):
        super().__init__(rng)
        self.mean = np.array(mean)
        self.cov = np.array(cov)

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
        self._data_shape = self._mean.shape
        self._data_size = self._mean.size

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, cov):
        self._cov = np.array(cov)
        if self._cov.shape != self._data_shape * 2:
            raise ValueError(f"Covariance array shape must be {self._data_shape * 2}.")

        # self._psd = _PSD(self._cov)
        # self._log_pf_coef = -0.5 * (self._psd.rank * np.log(2 * np.pi) + self._psd.log_pdet)

        self._inv_cov = inverse(self._cov)
        _log_det_cov = np.log(determinant(self._cov))
        self._log_pf_coef = -0.5 * (self._data_size * np.log(2 * np.pi) + _log_det_cov)

    @property
    def mode(self):
        return self.mean

    def _rvs(self, size=()):
        if self._data_shape == ():
            return self.rng.normal(self._mean, np.sqrt(self._cov), size).reshape(size + self._data_shape)
        else:
            return self.rng.multivariate_normal(self._mean, self._cov, size).reshape(size + self._data_shape)

    def pf(self, x):
        x, set_shape = check_data_shape(x, self._data_shape)

        dev = x.reshape(-1, *self._data_shape) - self._mean
        inner_white = np.array([(outer_gen(dev_i, np.ones(self._data_shape)) * self._inv_cov
                                * outer_gen(np.ones(self._data_shape), dev_i)).sum() for dev_i in dev])
        # inner_white = np.sum(np.square(np.dot(dev, self._psd.U)), axis=-1)

        log_pf = self._log_pf_coef + -0.5 * inner_white.reshape(set_shape)
        return np.exp(log_pf)

    def plot_pf(self, x_plt=None, ax=None):
        # _delta = 0.01
        n_plt = 100

        if self._data_size == 1:
            if x_plt is None:
                lims = self._mean.item() + np.array([-1, 1]) * 3*np.sqrt(self._cov.item())
                # n_plt = int(round((lims[1]-lims[0]) / _delta))
                x_plt = np.linspace(*lims, n_plt, endpoint=True).reshape(n_plt, *self._data_shape)

            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x_1$', ylabel='$p$')

            plt_data = ax.plot(x_plt, self.pf(x_plt))
            return plt_data

        elif self._data_size == 2:
            if x_plt is None:
                lims = [(self._mean[i] - 3 * np.sqrt(self._cov[i, i]), self._mean[i] + 3 * np.sqrt(self._cov[i, i]))
                        for i in range(2)]
                # n_plt = int(round((lims[0][1] - lims[0][0]) / _delta)), int(round((lims[1][1] - lims[1][0]) / _delta))
                x0_plt = np.linspace(*lims[0], n_plt, endpoint=True)
                x1_plt = np.linspace(*lims[1], n_plt, endpoint=True)
                x_plt = np.stack(np.meshgrid(x0_plt, x1_plt), axis=-1)

            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$p$')

            # ax.plot_wireframe(x_plt[..., 0], x_plt[..., 1], self.pf(x_plt))
            plt_data = ax.plot_surface(x_plt[..., 0], x_plt[..., 1], self.pf(x_plt), cmap=plt.cm.viridis)

            return plt_data
        else:
            raise NotImplementedError('Plot method only supported for 1- and 2-dimensional data.')


# # mean_, cov_ = np.ones(1), np.eye(1)
# mean_, cov_ = np.ones(2), np.eye(2)
# # mean_, cov_ = 1, 1
# norm = NormalRV(mean_, cov_)
# norm.rvs(5)
# norm.plot_pf()
