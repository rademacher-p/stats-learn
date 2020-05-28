"""
Random element objects.
"""

# TODO: docstrings?

import numpy as np
from scipy.stats._multivariate import multi_rv_generic
from scipy.special import gammaln, xlogy
import matplotlib.pyplot as plt
from util.generic import check_data_shape, check_valid_pmf
from util.math import outer_gen, diag_gen, simplex_round
from util.plotters import simplex_grid


#%% Base RE classes

class BaseRE(multi_rv_generic):
    """
    Base class for generic random element objects.
    """

    def __init__(self, rng=None):
        super().__init__(rng)      # may be None or int for legacy numpy rng

        self._data_shape = None
        self._mode = None

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def mode(self):
        return self._mode

    def rvs(self, size=(), random_state=None):
        if type(size) is int:
            size = (size,)
        elif type(size) is not tuple:
            raise TypeError("Input 'size' must be int or tuple.")
        random_state = self._get_random_state(random_state)

        return self._rvs(size, random_state)

    def _rvs(self, size=(), random_state=None):
        raise NotImplementedError("Method must be overwritten.")
        pass


class BaseRV(BaseRE):
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


class DiscreteRE(BaseRE):
    """
    Base class for discrete random element objects.
    """

    def pmf(self, x):
        x, set_shape = check_data_shape(x, self._data_shape)
        return self._pmf(x).reshape(set_shape)

    def _pmf(self, x):
        _out = []
        for x_i in x.reshape((-1,) + self._data_shape):
            _out.append(self._pmf_single(x_i))
        return np.asarray(_out)         # returned array may be flattened over 'set_shape'

    def _pmf_single(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass


class DiscreteRV(DiscreteRE, BaseRV):
    """
    Base class for discrete random variable (numeric) objects.
    """


class ContinuousRV(BaseRV):
    """
    Base class for continuous random element objects.
    """

    def pdf(self, x):
        x, set_shape = check_data_shape(x, self._data_shape)
        return self._pdf(x).reshape(set_shape)

    def _pdf(self, x):
        _out = []
        for x_i in x.reshape((-1,) + self._data_shape):
            _out.append(self._pdf_single(x_i))
        return np.asarray(_out)     # returned array may be flattened

    def _pdf_single(self, x):
        raise NotImplementedError("Method must be overwritten.")
        pass


#%% Specific RE's

class DeterministicRE(DiscreteRE):
    """
    Deterministic random element.
    """

    # TODO: redundant, just use FiniteRE? or change to ContinuousRV for integration?

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

    def _rvs(self, size=(), random_state=None):
        return np.broadcast_to(self._val, size + self._data_shape)

    def _pmf(self, x):
        return np.where(np.all(x.reshape(-1, self._data_size) == self._val.flatten(), axis=-1), 1., 0.)


class DeterministicRV(DeterministicRE, DiscreteRV):
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
# b.pmf(b.rvs())


class FiniteRE(DiscreteRE):
    """
    Generic RE drawn from a finite support set using an explicitly defined PMF.
    """

    def __new__(cls, supp, p, rng=None):
        supp = np.asarray(supp)
        if np.issubdtype(supp.dtype, np.number):
            return super().__new__(FiniteRV)
        else:
            return super().__new__(cls)

    def __init__(self, supp, p, rng=None):
        super().__init__(rng)
        self._supp = np.asarray(supp)
        self._p = check_valid_pmf(p)
        self._update_attr()

    # Input properties
    @property
    def supp(self):
        return self._supp

    @supp.setter
    def supp(self, supp):
        self._supp = np.asarray(supp)
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
        set_shape = self._supp.shape[:self._p.ndim]
        self._data_shape = self._supp.shape[self._p.ndim:]

        if set_shape != self._p.shape:
            raise ValueError("Leading shape values of 'supp' must equal the shape of 'p'.")

        self._supp_flat = self._supp.reshape((self._p.size, -1))
        self._p_flat = self._p.flatten()
        if len(self._supp_flat) != len(np.unique(self._supp_flat, axis=0)):
            raise ValueError("Input 'supp' must have unique values")

        self._mode = self._supp_flat[np.argmax(self._p_flat)].reshape(self._data_shape)

    def _rvs(self, size=(), random_state=None):
        i = random_state.choice(self.p.size, size, p=self._p_flat)
        return self._supp_flat[i].reshape(size + self._data_shape)

    def _pmf_single(self, x):
        eq_supp = np.all(x.flatten() == self._supp_flat, axis=-1)
        if eq_supp.sum() != 1:
            raise ValueError("Input 'x' must be in the support.")

        return self._p_flat[eq_supp].squeeze()

    def plot_pmf(self, ax=None):
        if self._p.ndim == 1:
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x$', ylabel=r'$\mathrm{P}_\mathrm{x}(x)$')

            plt_data = ax.stem(self._supp, self._p, use_line_collection=True)

            return plt_data
        else:
            raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')


class FiniteRV(FiniteRE, DiscreteRV):
    """
    Generic RV drawn from a finite support set using an explicitly defined PMF.
    """

    def _update_attr(self):
        super()._update_attr()

        mean_flat = (self._p_flat[:, np.newaxis] * self._supp_flat).sum(axis=0)
        self._mean = mean_flat.reshape(self._data_shape)

        ctr_flat = self._supp_flat - mean_flat
        outer_flat = (ctr_flat.reshape(self._p.size, 1, -1) * ctr_flat[..., np.newaxis]).reshape(self._p.size, -1)
        self._cov = (self._p_flat[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self._data_shape)

    def plot_pmf(self, ax=None):
        if self._p.ndim == 1:
            super().plot_pmf(self, ax)

        elif self._p.ndim in [2, 3]:
            if self._p.ndim == 2:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel=r'$\mathrm{P}_\mathrm{x}(x)$')

                plt_data = ax.bar3d(self._supp[0].flatten(), self._supp[1].flatten(), 0, 1, 1, self._p_flat, shade=True)

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


# s = np.random.random((4, 3, 2, 2))
# pp = np.random.random((4, 3))
# pp = pp / pp.sum()
# f = FiniteRE(s, pp)
# f.pmf(f.rvs((4,5)))
#
# s = np.stack(np.meshgrid([0,1],[0,1], [0,1]), axis=-1)
# s, p = ['a','b','c'], [.3,.2,.5]
# # p = np.random.random((2,2,2))
# # p = p / p.sum()
# f2 = FiniteRE(s, p)
# f2.pmf(f2.rvs(4))
# f2.plot_pmf()



def _dirichlet_check_alpha_0(alpha_0):
    alpha_0 = np.asarray(alpha_0)
    if alpha_0.size > 1 or alpha_0 <= 0:
        raise ValueError("Concentration parameter must be a positive scalar.")
    return alpha_0


def _dirichlet_check_input(x, alpha_0, mean):
    x = check_valid_pmf(x, data_shape=mean.shape)

    if np.logical_and(x == 0, mean < 1 / alpha_0).any():
        raise ValueError("Each entry in 'x' must be greater than zero if its mean is less than 1 / alpha_0.")

    return x


class DirichletRV(ContinuousRV):
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

        self._log_pdf_coef = gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))

    def _rvs(self, size=(), random_state=None):
        return random_state.dirichlet(self._alpha_0 * self._mean.flatten(), size).reshape(size + self._data_shape)

    def _pdf(self, x):
        x = _dirichlet_check_input(x, self._alpha_0, self._mean)

        log_pdf = self._log_pdf_coef + np.sum(xlogy(self._alpha_0 * self._mean - 1, x).reshape(-1, self._data_size), -1)
        return np.exp(log_pdf)

    def plot_pdf(self, n_plt, ax=None):

        if self._data_size in (2, 3):
            x_plt = simplex_grid(n_plt, self._data_shape, hull_mask=(self.mean < 1 / self.alpha_0))
            pdf_plt = self.pdf(x_plt)
            x_plt.resize(x_plt.shape[0], self._data_size)

            # pdf_plt.sum() / (n_plt ** (self._data_size - 1))

            if self._data_size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pdf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{p}_\mathrm{x}(x)$')

            elif self._data_size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.view_init(35, 45)
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], x_plt[:, 2], s=15, c=pdf_plt)

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
# d.plot_pdf(80)
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pdf(d.rvs())
# d.pdf(d.rvs(4).reshape((2, 2)+d.mean.shape))



def _empirical_check_n(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError("Input 'n' must be a positive integer.")
    return n


def _empirical_check_input(x, n, mean):
    x = check_valid_pmf(x, data_shape=mean.shape)

    # if ((n * x) % 1 > 0).any():
    if (np.minimum((n * x) % 1, (-n * x) % 1) > 1e-9).any():
        raise ValueError("Each entry in 'x' must be a multiple of 1/n.")

    return x


class EmpiricalRV(DiscreteRV):
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

        self._mode = ((self._n * self._mean) // 1) + simplex_round((self._n * self._mean) % 1)

        self._cov = (diag_gen(self._mean) - outer_gen(self._mean, self._mean)) / self._n

        self._log_pmf_coef = gammaln(self._n + 1)

    def _rvs(self, size=(), random_state=None):
        return random_state.multinomial(self._n, self._mean.flatten(), size).reshape(size + self._data_shape) / self._n

    def _pmf(self, x):
        x = _empirical_check_input(x, self._n, self._mean)

        log_pmf = self._log_pmf_coef + (xlogy(self._n * x, self._mean)
                                        - gammaln(self._n * x + 1)).reshape(-1, self._data_size).sum(axis=-1)
        return np.exp(log_pmf)

    def plot_pmf(self, ax=None):

        if self._data_size in (2, 3):
            x_plt = simplex_grid(self.n, self._data_shape)
            pmf_plt = self.pmf(x_plt)
            x_plt.resize(x_plt.shape[0], self._data_size)

            if self._data_size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pmf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

            elif self._data_size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.view_init(35, 45)
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], x_plt[:, 2], s=15, c=pmf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')

# rng = np.random.default_rng()
# n = 10
# m = np.random.random((1, 3))
# m = m / m.sum()
# d = EmpiricalRV(n, m, rng)
# d.plot_pmf()
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pmf(d.rvs())
# d.pmf(d.rvs(4).reshape((2, 2) + d.mean.shape))


class DirichletEmpiricalRV(DiscreteRV):
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

        self._log_pmf_coef = (gammaln(self._alpha_0) - np.sum(gammaln(self._alpha_0 * self._mean))
                              + gammaln(self._n + 1) - gammaln(self._alpha_0 + self._n))

    def _rvs(self, size=(), random_state=None):
        return random_state.multinomial(self._n, self._mean.flatten(), size).reshape(size + self._data_shape) / self._n

    def _pmf(self, x):
        x = _empirical_check_input(x, self._n, self._mean)

        log_pmf = self._log_pmf_coef + (gammaln(self._alpha_0 * self._mean + self._n * x)
                                        - gammaln(self._n * x + 1)).reshape(-1, self._data_size).sum(axis=-1)
        return np.exp(log_pmf)

    def plot_pmf(self, ax=None):        # TODO: reused code. define simplex plotter outside!

        if self._data_size in (2, 3):
            x_plt = simplex_grid(self.n, self._data_shape)
            pmf_plt = self.pmf(x_plt)
            x_plt.resize(x_plt.shape[0], self._data_size)

            if self._data_size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pmf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

            elif self._data_size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.view_init(35, 45)
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], x_plt[:, 2], s=15, c=pmf_plt)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label(r'$\mathrm{P}_\mathrm{x}(x)$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')

# rng = np.random.default_rng()
# n = 10
# a0 = 600
# m = np.ones((1, 3))
# m = m / m.sum()
# d = DirichletEmpiricalRV(n, a0, m, rng)
# d.plot_pmf()
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pmf(d.rvs())
# d.pmf(d.rvs(4).reshape((2, 2) + d.mean.shape))
