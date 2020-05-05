"""
Random element objects.
"""

# TODO: setters allow RE/RV type switching. Remove?
# TODO: change rvs method name to RE?
# TODO: docstrings?

import numpy as np
from scipy.stats._multivariate import multi_rv_generic
from scipy.special import gammaln, xlogy
import matplotlib.pyplot as plt
from util.util import outer_gen, diag_gen, simplex_grid


def _check_data_shape(x, shape):
    """Checks input shape for RV cdf/pdf calls"""

    x = np.asarray(x)

    if x.shape == shape:
        set_shape = ()
    elif shape == ():
        set_shape = x.shape
    elif x.shape[-len(shape):] == shape:
        set_shape = x.shape[:-len(shape)]
    else:
        raise TypeError("Trailing dimensions of 'shape' must be equal to the shape of 'x'.")

    return x, set_shape


def _vectorize_func(func, data_shape):
    def func_vec(x):
        x, set_shape = _check_data_shape(x, data_shape)

        _out = []
        for x_i in x.reshape((-1,) + data_shape):
            _out.append(func(x_i))
        _out = np.asarray(_out)
        return _out.reshape(set_shape + _out.shape[1:])

    return func_vec


def _check_valid_pmf(p, shape=None, full_support=False):
    if shape is None:
        p = np.asarray(p)
        set_shape = ()
    else:
        p, set_shape = _check_data_shape(p, shape)

    if full_support:
        if np.min(p) <= 0:
            raise ValueError("Each entry in 'p' must be greater than zero.")
    else:
        if np.min(p) < 0:
            raise ValueError("Each entry in 'p' must be greater than or equal to zero.")

    if (np.abs(p.reshape(set_shape + (-1,)).sum(-1) - 1.0) > 1e-9).any():
        raise ValueError("The input 'p' must lie within the normal simplex, but p.sum() = %s." % p.sum())

    return p


#%% Base RE classes

class GenericRE(multi_rv_generic):
    """
    Base class for generic random element objects.
    """

    def __init__(self, seed=None):
        super().__init__(seed)

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
        return None


class GenericRV(GenericRE):
    """
    Base class for generic random variable (numeric) objects.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self._mean = None
        self._cov = None

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov


class DiscreteRE(GenericRE):
    """
    Base class for discrete random element objects.
    """

    def pmf(self, x):
        x, set_shape = _check_data_shape(x, self._data_shape)
        return self._pmf(x).reshape(set_shape)

    def _pmf(self, x):
        _out = []
        for x_i in x.reshape((-1,) + self._data_shape):
            _out.append(self._pmf_single(x_i))
        return np.asarray(_out)         # returned array may be flattened over 'set_shape'

    def _pmf_single(self, x):
        return None


class DiscreteRV(DiscreteRE, GenericRV):
    """
    Base class for discrete random variable (numeric) objects.
    """


class ContinuousRV(GenericRV):
    """
    Base class for continuous random element objects.
    """

    def pdf(self, x):
        x, set_shape = _check_data_shape(x, self._data_shape)
        return self._pdf(x).reshape(set_shape)

    def _pdf(self, x):
        _out = []
        for x_i in x.reshape((-1,) + self._data_shape):
            _out.append(self._pdf_single(x_i))
        return np.asarray(_out)     # returned array may be flattened

    def _pdf_single(self, x):
        return None


#%% Specific RE's

class DeterministicRE(DiscreteRE):
    """
    Deterministic random element.
    """

    # TODO: redundant, just use FiniteRE? or change to ContinuousRV for integration?

    def __new__(cls, val, seed=None):
        val = np.asarray(val)
        if np.issubdtype(val.dtype, np.number):
            return super().__new__(DeterministicRV)
        else:
            return super().__new__(cls)

    def __init__(self, val, seed=None):
        super().__init__(seed)
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

    # def _pmf_single(self, x):
    #     return 1. if (x == self.val).all() else 0.


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

    # TODO: reconsider support shape?

    def __new__(cls, supp, p, seed=None):
        supp = np.asarray(supp)
        if np.issubdtype(supp.dtype, np.number):
            return super().__new__(FiniteRV)
        else:
            return super().__new__(cls)

    def __init__(self, supp, p, seed=None):
        super().__init__(seed)
        self._update_attr(supp, p)

    # Input properties
    @property
    def supp(self):
        return self._supp

    @supp.setter
    def supp(self, supp):
        self._update_attr(supp=supp)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._update_attr(p=p)

    # Attribute Updates
    def _update_attr(self, *args, **kwargs):
        if 'supp' in kwargs.keys():
            self._supp = np.asarray(kwargs['supp'])
        elif len(args) > 0:
            self._supp = np.asarray(args[0])

        if 'p' in kwargs.keys():
            self._p = _check_valid_pmf(kwargs['p'])
        elif len(args) > 1:
            self._p = _check_valid_pmf(args[1])

        set_shape = self._supp.shape[:self._p.ndim]
        self._data_shape = self._supp.shape[self._p.ndim:]

        if set_shape != self._p.shape:
            raise ValueError("Leading shape values of 'supp' must equal the shape of 'p'.")

        self._supp_flat = self._supp.reshape((self._p.size, -1))        # TODO: order???
        self._p_flat = self._p.flatten()
        if len(self._supp_flat) != len(np.unique(self._supp_flat, axis=0)):     # TODO: axis???
            raise ValueError("Input 'supp' must have unique values")

        self._mode = self._supp_flat[np.argmax(self._p_flat)].reshape(self._data_shape)

    def _rvs(self, size=(), random_state=None):
        i = random_state.choice(self.p.size, size, p=self._p_flat)
        return self._supp_flat[i].reshape(size + self._data_shape)

    def _pmf_single(self, x):
        eq_supp = np.all(x.flatten() == self._supp_flat, axis=-1)
        if eq_supp.sum() != 1:
            raise ValueError("Input 'x' must be lie in the support.")

        return self._p_flat[eq_supp]

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

    def _update_attr(self, *args, **kwargs):
        super()._update_attr(*args, **kwargs)

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
# f.pmf(f.rvs(4))
#
# s = np.stack(np.meshgrid([0,1],[0,1], [0,1]), axis=-1)
# s, p = ['a','b','c'], [.3,.2,.5]
# # p = np.random.random((2,2,2))
# # p = p / p.sum()
# f2 = FiniteRE(s, p)
# f2.pmf(f2.rvs(4))
# f2.plot_pmf()



def _dirichlet_check_alpha_0(alpha_0):      # TODO: delete? implicit type checking in subsequent code
    # alpha_0 = np.asarray(alpha_0)
    # if alpha_0.size > 1 or alpha_0 <= 0:
    #     raise ValueError("Concentration parameter must be a positive scalar.")
    return alpha_0


def _dirichlet_check_input(x, alpha_0, mean):
    _check_valid_pmf(x, shape=mean.shape)

    if np.logical_and(x == 0, mean < 1 / alpha_0).any():
        raise ValueError("Each entry in 'x' must be greater than zero if its mean is less than 1 / alpha_0.")

    return x


class DirichletRV(ContinuousRV):
    """
    Dirichlet random process, finite-domain realizations.
    """

    def __init__(self, alpha_0, mean, seed=None):
        super().__init__(seed)
        self._update_attr(alpha_0, mean)

    # Input properties
    @property
    def alpha_0(self):
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._update_attr(alpha_0=alpha_0)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._update_attr(mean=mean)

    # Attribute Updates
    def _update_attr(self, *args, **kwargs):

        if 'alpha_0' in kwargs.keys():
            self._alpha_0 = _dirichlet_check_alpha_0(kwargs['alpha_0'])
        elif len(args) > 0:
            self._alpha_0 = _dirichlet_check_alpha_0(args[0])

        if 'mean' in kwargs.keys():
            self._mean = _check_valid_pmf(kwargs['mean'], full_support=True)
        elif len(args) > 1:
            self._mean = _check_valid_pmf(args[1], full_support=True)

        self._data_shape = self._mean.shape
        self._data_size = self._mean.size

        if np.min(self.mean) > 1 / self.alpha_0:
            self._mode = (self.mean - 1 / self.alpha_0) / (1 - self._data_size / self.alpha_0)
        else:
            # warnings.warn("Mode method currently supported for mean > 1/alpha_0 only")
            self._mode = None       # TODO: complete with general formula

        self._cov = (diag_gen(self.mean) - outer_gen(self.mean, self.mean)) / (self.alpha_0 + 1)

        self._beta_inv = gammaln(np.sum(self.alpha_0 * self.mean)) - np.sum(gammaln(self.alpha_0 * self.mean))

    def _rvs(self, size=(), random_state=None):
        return random_state.dirichlet(self.alpha_0 * self.mean.flatten(), size).reshape(size + self._data_shape)

    def _pdf(self, x):
        x = _dirichlet_check_input(x, self.alpha_0, self.mean)

        log_pdf = self._beta_inv + np.sum(xlogy(self.alpha_0 * self.mean - 1, x).reshape(-1, self._data_size), -1)
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


rng = np.random.default_rng()
a0 = 4
m = np.random.random((1, 3))
m = m / m.sum()
d = DirichletRV(a0, m, rng)
d.plot_pdf(30)
d.mean
d.mode
d.cov
d.rvs()
d.pdf(d.rvs())
d.pdf(d.rvs(4).reshape((2, 2)+d.mean.shape))



def _empirical_check_n(n):      # TODO: delete? implicit type checking in subsequent code
    return n


def _empirical_check_input(x, n, mean):
    _check_valid_pmf(x, shape=mean.shape)

    # TODO: incomplete, check int

    if np.logical_and(x == 0, mean < 1 / alpha_0).any():
        raise ValueError("Each entry in 'x' must be greater than zero if its mean is less than 1 / alpha_0.")

    return x


class EmpiricalRE(DiscreteRV):
    """
    Empirical random process, finite-domain realizations.
    """

    def __init__(self, n, mean, seed=None):
        super().__init__(seed)
        self._update_attr(n, mean)

    # Input properties
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._update_attr(n=n)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._update_attr(mean=mean)

    # Attribute Updates
    def _update_attr(self, *args, **kwargs):

        if 'n' in kwargs.keys():
            self._n = _empirical_check_n(kwargs['n'])
        elif len(args) > 0:
            self._n = _empirical_check_n(args[0])

        if 'mean' in kwargs.keys():
            self._mean = _check_valid_pmf(kwargs['mean'])
        elif len(args) > 1:
            self._mean = _check_valid_pmf(args[1])

        self._data_shape = self._mean.shape
        self._data_size = self._mean.size

        if np.min(self.mean) > 1 / self.alpha_0:
            self._mode = (self.mean - 1 / self.alpha_0) / (1 - self._data_size / self.alpha_0)
        else:
            # warnings.warn("Mode method currently supported for mean > 1/alpha_0 only")
            self._mode = None       # TODO: complete with general formula

        self._cov = (diag_gen(self.mean) - outer_gen(self.mean, self.mean)) / self.n

        self._beta_inv = gammaln(np.sum(self.alpha_0 * self.mean)) - np.sum(gammaln(self.alpha_0 * self.mean))

    def _rvs(self, size=(), random_state=None):
        return random_state.dirichlet(self.alpha_0 * self.mean.flatten(), size).reshape(size + self._data_shape)

    def _pdf(self, x):
        x = _dirichlet_check_input(x, self.alpha_0, self.mean)

        log_pdf = self._beta_inv + np.sum(xlogy(self.alpha_0 * self.mean - 1, x).reshape(-1, self._data_size), -1)
        return np.exp(log_pdf)

