"""
Random element objects
"""

# TODO: add numeric subclasses of BaseRE?
# TODO: add placeholder methods for pdf, pdf_single, etc.?
# TODO: docstrings?


import warnings

import numpy as np
from scipy.stats._multivariate import multi_rv_generic
from scipy.special import gammaln, xlogy
import matplotlib.pyplot as plt

from util.util import outer_gen, diag_gen, simplex_grid


def _check_data_shape(x, shape):
    """Checks input shape for RV cdf/pdf calls"""

    x = np.asarray(x)

    # if len(shape) > 0 and x.shape[-len(shape):] != shape:
    #     raise TypeError("Trailing dimensions of 'shape' must be equal to the shape of 'x'.")
    # return x

    if x.shape == shape:
        set_shape = ()
    elif len(shape) == 0:
        set_shape = x.shape
    elif x.shape[-len(shape):] == shape:
        set_shape = x.shape[:-len(shape)]
    else:
        raise TypeError("Trailing dimensions of 'shape' must be equal to the shape of 'x'.")

    return x, set_shape


class BaseRE(multi_rv_generic):
    def __init__(self, *args, seed=None, **kwargs):
        # super().__init__(kwargs['seed'])
        super().__init__(seed)
        self._update_attr(*args, **kwargs)

    def _update_attr(self, *args, **kwargs):
        self._data_shape = None
        self._data_size = None

        self._mode = None
        self._mean = None
        self._cov = None

    @property
    def mode(self):
        return self._mode

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    def rvs(self, size=(), random_state=None):
        if type(size) is int:
            size = (size,)
        random_state = self._get_random_state(random_state)
        return self._rvs(size, random_state)

    def _rvs(self, size=(), random_state=None):
        return None


#%% Deterministic RV, multivariate

# TODO: redundant, just use FiniteRE? or continuous domain version?

class DeterministicRE(BaseRE):
    def __init__(self, val, seed=None):
        super().__init__(val, seed=seed)

    # Input properties
    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._update_attr(val)

    # Base method overwrites
    def _update_attr(self, val):
        val = np.asarray(val)
        self._val = val

        self._data_shape = self._val.shape
        self._data_size = self._val.size

        self._mode = self._val
        if np.issubdtype(self._val.dtype, np.number):
            self._mean = self._val
            self._cov = np.zeros(2 * self._data_shape)
        else:
            # warnings.warn("Method only supported for numeric 'val'.")
            self._mean = None
            self._cov = None

    def _rvs(self, size=(), random_state=None):
        return np.broadcast_to(self.val, size + self._data_shape)

    def pmf(self, x):
        x, set_shape = _check_data_shape(x, self._data_shape)
        return np.where(np.all(x.reshape(-1, self._data_size) == self.val.flatten(), axis=-1), 1., 0.).reshape(set_shape)



# rng = np.random.default_rng()
# a = np.arange(6).reshape(3, 2)
# # a = ['a','b','c']
# b = DeterministicRE(a[1], rng)
# b.mode
# b.mean
# b.cov
# b.pmf(b.rvs())



#%% Discrete RV, multivariate (generalized)

# def _discrete_check_parameters(supp, p):
#     supp = np.asarray(supp)
#     p = np.asarray(p)
#
#     set_shape, data_shape = supp.shape[:p.ndim], supp.shape[p.ndim:]
#     if set_shape != p.shape:
#         raise ValueError("Leading shape values of 'supp' must equal the shape of 'pmf'.")
#
#     supp_flat = supp.reshape((np.prod(set_shape), -1))
#     if len(supp_flat) != len(np.unique(supp_flat, axis=0)):
#         raise ValueError("Input 'supp' must have unique values")
#
#     if np.min(p) < 0:
#         raise ValueError("Each entry in 'pmf' must be greater than or equal "
#                          "to zero.")
#     if np.abs(p.sum() - 1.0) > 1e-9:
#         raise ValueError("The input 'pmf' must lie within the normal "
#                          "simplex. but pmf.sum() = %s." % p.sum())
#
#     return supp, p


class FiniteRE(BaseRE):

    # TODO: add plot methods!

    def __init__(self, supp, p, seed=None):
        super().__init__(supp, p, seed=seed)

    # Input properties
    @property
    def supp(self):
        return self._supp

    @supp.setter
    def supp(self, supp):
        self._update_attr(supp, self._p)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._update_attr(self._supp, p)

    # Base method overwrites
    def _update_attr(self, supp, p):
        # self._supp, self._p = _discrete_check_parameters(supp, p)
        self._supp = np.asarray(supp)
        self._p = np.asarray(p)

        set_shape = self._supp.shape[:self._p.ndim]
        self._data_shape = self._supp.shape[self._p.ndim:]

        if set_shape != self._p.shape:
            raise ValueError("Leading shape values of 'supp' must equal the shape of 'p'.")

        self._supp_flat = self._supp.reshape((self.p.size, -1))
        self._p_flat = self.p.flatten()
        if len(self._supp_flat) != len(np.unique(self._supp_flat, axis=0)):
            raise ValueError("Input 'supp' must have unique values")

        if np.min(self._p) < 0:
            raise ValueError("Each entry in 'p' must be greater than or equal "
                             "to zero.")
        if np.abs(self._p.sum() - 1.0) > 1e-9:
            raise ValueError("The input 'pmf' must lie within the normal "
                             "simplex. but p.sum() = %s." % self._p.sum())

        self._mode = self._supp_flat[np.argmax(self._p_flat)].reshape(self._data_shape)

        if np.issubdtype(self.supp.dtype, np.number):
            mean_flat = (self._p_flat[:, np.newaxis] * self._supp_flat).sum(axis=0)
            self._mean = mean_flat.reshape(self._data_shape)
            ctr_flat = self._supp_flat - mean_flat
            outer_flat = (ctr_flat.reshape(self.p.size, 1, -1) * ctr_flat[..., np.newaxis]).reshape(self.p.size, -1)
            self._cov = (self._p_flat[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self._data_shape)
        else:
            # warnings.warn("Method only supported for numeric 'supp'.")
            self._mean = None
            self._cov = None

    def _rvs(self, size=(), random_state=None):
        i = random_state.choice(self.p.size, size, p=self._p_flat)
        return self._supp_flat[i].reshape(size + self._data_shape)

    def pmf(self, x):
        x, set_shape = _check_data_shape(x, self._data_shape)

        _out = []
        for x_i in x.reshape(int(np.prod(set_shape)), -1):
            _out.append(self._p_flat[np.all(x_i == self._supp_flat, axis=-1)])
        return np.asarray(_out).reshape(set_shape)

    def plot_pmf(self, ax=None):
        if self._p.ndim == 1:
            if ax is None:
                _, ax = plt.subplots()

            plt_data = ax.stem(self._p, use_line_collection=True)
            ax.set_xticks(range(self._p.size))
            ax.set_xticklabels()

        return plt_data

        # _, ax_theta = plt.subplots(num='theta pmf', clear=True, subplot_kw={'projection': '3d'})
        # # ax_theta.scatter(YX_set['x'], YX_set['y'], theta_pmf, c=theta_pmf)
        # ax_theta.bar3d(YX_set['x'].flatten(), YX_set['y'].flatten(), 0, 1, 1, theta_pmf.flatten(), shade=True)
        # ax_theta.set(xlabel='$x$', ylabel='$y$')




s = np.random.random((4, 3, 2, 1))
pp = np.random.random((4, 3))
pp = pp / pp.sum()
f = FiniteRE(s, pp)
f.pmf(f.rvs())



#%% Dirichlet RV, multivariate (generalized dimension)

def _dirichlet_check_alpha_0(alpha_0):
    alpha_0 = np.asarray(alpha_0)
    if alpha_0.size > 1 or alpha_0 <= 0:
        raise ValueError("Concentration parameter must be a positive scalar")
    return alpha_0


def _dirichlet_check_mean(mean):
    mean = np.asarray(mean)
    if np.min(mean) < 0:
        raise ValueError("Each entry in 'mean' must be greater than or equal "
                         "to zero.")
    if np.abs(mean.sum() - 1.0) > 1e-9:
        raise ValueError("The input 'mean' must lie within the normal "
                         "simplex. but mean.sum() = %s." % mean.sum())
    return mean


def _dirichlet_multi_check_input(x, alpha_0, mean):
    x, set_shape = _check_data_shape(x, mean.shape)

    if np.min(x) < 0:
        raise ValueError("Each entry in 'x' must be greater than or equal "
                         "to zero.")

    if (np.abs(x.reshape(-1, mean.size).sum(-1) - 1.0) > 1e-9).any():
        raise ValueError("The sample values in 'x' must lie within the normal "
                         "simplex, but the sums = %s." % x.reshape(-1, mean.size).sum(-1).squeeze())

    if np.logical_and(x == 0, mean < 1 / alpha_0).any():
        raise ValueError("Each entry in 'x' must be greater than zero if its "
                         "mean is less than 1 / alpha_0.")

    return x, set_shape


class DirichletRE(BaseRE):
    def __init__(self, alpha_0, mean, seed=None):
        super().__init__(alpha_0, mean, seed=seed)

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

    # Base method overwrites
    def _update_attr(self, *args, **kwargs):
        if 'alpha_0' in kwargs.keys():
            self._alpha_0 = _dirichlet_check_alpha_0(kwargs['alpha_0'])
        elif len(args) > 0:
            self._alpha_0 = _dirichlet_check_alpha_0(args[0])

        if 'mean' in kwargs.keys():
            self._mean = _dirichlet_check_mean(kwargs['mean'])
        elif len(args) > 1:
            self._mean = _dirichlet_check_mean(args[1])

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

    def pdf(self, x):
        x, set_shape = _dirichlet_multi_check_input(x, self.alpha_0, self.mean)

        log_pdf = self._beta_inv + np.sum(xlogy(self.alpha_0 * self.mean - 1, x).reshape(-1, self._data_size), -1)
        return np.exp(log_pdf).reshape(set_shape)

    def plot_pdf(self, n_plt, ax=None):

        if self._data_size in (2, 3):
            x_plt = simplex_grid(n_plt, self._data_shape, hull_mask=(self.mean < 1 / self.alpha_0))
            p_theta_plt = self.pdf(x_plt)
            x_plt.resize(x_plt.shape[0], self._data_size)

            # p_theta_plt.sum() / (n_plt ** (self._data_size - 1))

            if self._data_size == 2:
                if ax is None:
                    _, ax = plt.subplots()

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=p_theta_plt)
                plt.colorbar(plt_data)
                ax.set(xlabel='$x_1$', ylabel='$x_2$')
            elif self._data_size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], x_plt[:, 2], s=15, c=p_theta_plt)
                ax.view_init(35, 45)
                plt.colorbar(plt_data)
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')




# a0 = 4
# m = np.random.random((3, 2))
# m = m / m.sum()
# d = DirichletRE(a0, m, rng)
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pdf(d.rvs())
# d.pdf(d.rvs(4).reshape((2, 2)+d.mean.shape))
