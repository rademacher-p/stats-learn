"""
Random element objects
"""

# TODO: add numeric subclasses of BaseRE? use __new__ to handle input parameter types?
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
    """
    Base class for random element objects.
    """

    def __init__(self, seed=None):
        super().__init__(seed)

        self._data_shape = None

        self._mode = None
        self._mean = None
        self._cov = None


    @property
    def data_shape(self):
        return self._data_shape

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


#%% Deterministic

# TODO: redundant, just use FiniteRE? or continuous domain version for integration?

class DeterministicRE(BaseRE):
    """
    Deterministic random element.
    """

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
        if np.issubdtype(self._val.dtype, np.number):
            self._mean = self._val
            self._cov = np.zeros(2 * self._data_shape)
        else:
            del self._mean, self._cov       # TODO: del? or super None?

    def _rvs(self, size=(), random_state=None):
        return np.broadcast_to(self._val, size + self._data_shape)

    def pmf(self, x):
        x, set_shape = _check_data_shape(x, self._data_shape)
        return np.where(np.all(x.reshape(-1, self._data_size) == self._val.flatten(), axis=-1), 1., 0.).reshape(set_shape)


# rng = np.random.default_rng()
# a = np.arange(6).reshape(3, 2)
# # a = ['a','b','c']
# b = DeterministicRE(a[1], rng)
# b.mode
# b.mean
# b.cov
# b.pmf(b.rvs())


#%% Generic, finite support

class FiniteRE(BaseRE):
    """
    Generic RE drawn from a finite support set using an explicitly defined PMF.
    """

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
            self._p = np.asarray(kwargs['p'])
        elif len(args) > 1:
            self._p = np.asarray(args[1])

        set_shape = self._supp.shape[:self._p.ndim]
        self._data_shape = self._supp.shape[self._p.ndim:]

        if set_shape != self._p.shape:
            raise ValueError("Leading shape values of 'supp' must equal the shape of 'p'.")

        self._supp_flat = self._supp.reshape((self._p.size, -1))
        self._p_flat = self._p.flatten()
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
            outer_flat = (ctr_flat.reshape(self._p.size, 1, -1) * ctr_flat[..., np.newaxis]).reshape(self._p.size, -1)
            self._cov = (self._p_flat[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self._data_shape)
        else:
            del self._mean, self._cov   # del? or super None?

    def _rvs(self, size=(), random_state=None):
        i = random_state.choice(self.p.size, size, p=self._p_flat)
        return self._supp_flat[i].reshape(size + self._data_shape)

    def pmf(self, x):       # TODO: pmf single?
        x, set_shape = _check_data_shape(x, self._data_shape)

        _out = []
        for x_i in x.reshape(int(np.prod(set_shape)), -1):
            _out.append(self._p_flat[np.all(x_i == self._supp_flat, axis=-1)])
        return np.asarray(_out).reshape(set_shape)

    def plot_pmf(self, ax=None):
        # return None

        if self._p.ndim in [1, 2]:
            if self._p.ndim == 1:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set_xticks(range(self._p.size))
                    ax.set_xticklabels(self._supp)
                    ax.set(xlabel='$x$', ylabel='$\mathrm{P}_\mathrm{x}(x)$')

                plt_data = ax.stem(self._p, use_line_collection=True)

            elif self._p.ndim == 2:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.set_xticks(range(self._p.size))
                    ax.set_xticklabels(self._supp)
                    ax.set(xlabel='$x$', ylabel='$\mathrm{P}_\mathrm{x}(x)$')

                # plt_data = ax.bar3d(YX_set['x'].flatten(), YX_set['y'].flatten(), 0, 1, 1, self._p_flat, shade=True)
                # TODO: incomplete, add 3-d
            return plt_data

        else:
            raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')



# s = np.random.random((4, 3, 2, 1))
# pp = np.random.random((4, 3))
# pp = pp / pp.sum()
# f = FiniteRE(s, pp)
# f.pmf(f.rvs())

# f = FiniteRE(['a','b','c'], [.3,.2,.5])
# f.plot_pmf()



#%% Dirichlet

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
            pdf_plt = self.pdf(x_plt)
            x_plt.resize(x_plt.shape[0], self._data_size)

            # pdf_plt.sum() / (n_plt ** (self._data_size - 1))

            if self._data_size == 2:
                if ax is None:
                    _, ax = plt.subplots()
                    ax.set(xlabel='$x_1$', ylabel='$x_2$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], s=15, c=pdf_plt)

                cbar = plt.colorbar(plt_data)
                cbar.set_label('$\mathrm{p}_\mathrm{x}(x)$')

            elif self._data_size == 3:
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.view_init(35, 45)
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(x_plt[:, 0], x_plt[:, 1], x_plt[:, 2], s=15, c=pdf_plt)

                cbar = plt.colorbar(plt_data)
                cbar.set_label('$\mathrm{p}_\mathrm{x}(x)$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')


rng = np.random.default_rng()
a0 = 4
m = np.random.random((1,2))
m = m / m.sum()
d = DirichletRE(a0, m, rng)
d.plot_pdf(30)
# d.mean
# d.mode
# d.cov
# d.rvs()
# d.pdf(d.rvs())
# d.pdf(d.rvs(4).reshape((2, 2)+d.mean.shape))



#%% Supervised Learning classes

class ModelBase(BaseRE):
    """
    Base class for supervised learning data models.
    """

    def __init__(self, seed=None):
        super().__init__(seed)

        self._data_shape_x = None
        self._data_shape_y = None

        self._mode_x = None
        self._mode_y = None
        self._mode_x_y = None
        self._mode_y_x = None

        self._mean_x = None
        self._mean_y = None
        self._mean_x_y = None
        self._mean_y_x = None

        self._cov_x = None
        self._cov_y = None
        self._cov_x_y = None
        self._cov_y_x = None

    @property
    def data_shape_x(self):
        return self._data_shape_x

    @property
    def data_shape_y(self):
        return self._data_shape_y

    @property
    def mode_x(self):
        return self._mode_x

    @property
    def mode_y(self):
        return self._mode_y

    @property
    def mode_x_y(self):
        return self._mode_x_y

    def mode_y_x(self, x):      # TODO: as method?
        return None

    @property
    def mean_x(self):
        return self._mean_x

    @property
    def mean_y(self):
        return self._mean_y

    @property
    def mean_x_y(self):
        return self._mean_x_y

    @property
    def mean_y_x(self):
        return self._mean_y_x

    @property
    def cov_x(self):
        return self._cov_x

    @property
    def cov_y(self):
        return self._cov_y

    @property
    def cov_x_y(self):
        return self._cov_x_y

    @property
    def cov_y_x(self):
        return self._cov_y_x




class ModelCondX(ModelBase):
    def __init__(self, model_x, model_y_x, seed=None):
        super().__init__(seed)
        self._update_attr(model_x, model_y_x)

    @property
    def model_x(self):
        return self._model_x

    @model_x.setter
    def model_x(self, model_x):
        self._update_attr(model_x=model_x)

    @property
    def model_y_x(self):
        return self._model_y_x

    @model_y_x.setter
    def model_y_x(self, model_y_x):
        self._update_attr(model_y_x=model_y_x)

    def _update_attr(self, *args, **kwargs):

        if 'model_x' in kwargs.keys():
            self._model_x = kwargs['model_x']
            self._update_x()
        elif len(args) > 0:
            self._model_x = args[0]
            self._update_x()

        if 'model_y_x' in kwargs.keys():
            self._model_y_x = kwargs['model_y_x']
            self._update_y_x()
        elif len(args) > 1:
            self._model_y_x = args[1]
            self._update_y_x()

    def _update_x(self):
        self._data_shape_x = self._model_x.data_shape

        self._mode_x = self._model_x.mode
        self._mean_x = self._model_x.mean
        self._cov_x = self._model_x.cov

    def _update_y_x(self):
        self._data_shape_y = self._model_y_x(self._model_x.rvs()).data_shape

        # def _mode_y_x(self, x): return self._model_y_x(x).mode

        # self._mode_y_x = self._model_y_x(x_sample).mode
        # self._mean_y_x = self._model_y_x(x_sample).mean
        # self._cov_y_x = self._model_y_x(x_sample).cov

    def mode_y_x(self, x):              # TODO: vectorize?
        return self._model_y_x(x).mode



    def _rvs(self, size=(), random_state=None):
        X = np.asarray(self.model_x.rvs(size, random_state))
        if len(size) == 0:
            Y = self.model_y_x(X).rvs(size, random_state)
            D = np.array((Y, X), dtype=[('y', Y.dtype, self.data_shape_y), ('x', X.dtype, self.data_shape_x)])
        else:
            Y = np.asarray([self.model_y_x(x).rvs((), random_state) for x in X.reshape((-1,) + self.model_x._data_shape)])\
                .reshape(size + self.data_shape_y)
            D = np.array(list(zip(Y.reshape((-1,) + self.data_shape_y), X.reshape((-1,) + self.data_shape_x))),
                         dtype=[('y', Y.dtype, self.data_shape_y), ('x', X.dtype, self.data_shape_x)]).reshape(size)

        return D



model_x = DirichletRE(4, [.5, .5])
def model_y_x(x): return FiniteRE(['a', 'b'], x)

t = ModelCondX(model_x, model_y_x)
t.rvs()
t.rvs(4)
t.mode_y_x(t.model_x.rvs())
