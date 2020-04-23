"""
Sim main.

:-)
"""

import itertools

import numpy as np
from numpy import random
from scipy import stats
from scipy.stats._multivariate import multi_rv_generic
# from scipy._lib._util import check_random_state
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# from rv_obj import deterministic_multi, dirichlet_multi, discrete_multi
from rv_obj import BaseRE, DeterministicRE, DirichletRE, FiniteRE

# plt.style.use('seaborn')  # cm?

rng = random.default_rng()



#%% Continuous sets

# theta_m = stats.beta(a=.9, b=.9)
# def theta_c(x): return stats.beta(5*x, 5*(1-x))
#
#
# plt.figure(num='theta', clear=True)
#
# x_plot = np.linspace(0, 1, 101, endpoint=True)
# plt.subplot(1, 2, 1)
# plt.plot(x_plot, theta_m.pdf(x_plot))
# plt.gca().set(title='Marginal Model', xlabel='$x$', ylabel=r'$p_{\theta_m}(x)$')
# plt.gca().set_ylim(0)
#
# X = theta_m.rvs()
#
# y_plot = np.linspace(0, 1, 101, endpoint=True)
# plt.subplot(1, 2, 2)
# plt.plot(y_plot, theta_c(X).pdf(x_plot))
# plt.gca().set(title='Conditional Model', xlabel='$y$', ylabel=r'$p_{\theta_c}(y;x)$')
# plt.gca().set_ylim(0)
#
# Y = theta_c(X).rvs()
#
# plt.suptitle(f'Model, (X,Y) = ({X:.2f},{Y:.2f})')


theta_m = stats.multivariate_normal(mean=[0, 0])
def theta_c(x): return stats.multivariate_normal(mean=x)


_, ax_theta_m = plt.subplots(num='theta_m', clear=True, subplot_kw={'projection': '3d'})

x1_plot = np.linspace(-5, 5, 101, endpoint=True)
x2_plot = np.linspace(-5, 5, 51, endpoint=True)
X_plot = np.stack(np.meshgrid(x1_plot, x2_plot), axis=-1)

ax_theta_m.plot_wireframe(X_plot[..., 0], X_plot[..., 1], theta_m.pdf(X_plot))
# ax_prior[0].plot_wireframe(X_plot[0], X_plot[1], theta_m.pdf(t))
plt.gca().set(title='Marginal Model', xlabel='$x$', ylabel=r'$p_{\theta_m}(x)$')

X = theta_m.rvs()

_, ax_theta_c = plt.subplots(num='theta_c', clear=True, subplot_kw={'projection': '3d'})

y1_plot = np.linspace(-5, 5, 101, endpoint=True)
y2_plot = np.linspace(-5, 5, 51, endpoint=True)
Y_plot = np.stack(np.meshgrid(y1_plot, y2_plot), axis=-1)

ax_theta_c.plot_wireframe(Y_plot[..., 0], Y_plot[..., 1], theta_c(X).pdf(Y_plot))
plt.gca().set(title='Conditional Model', xlabel='$y$', ylabel=r'$p_{\theta_c}(y;x)$')

Y = theta_c(X).rvs()


#%% Discrete sets

Y_set = np.array(['a', 'b'])
# Y_set = np.arange(3)
# X_set = np.arange(1)
# X_set = np.arange(6).reshape(3, 2)
X_set = np.stack(np.meshgrid(np.arange(2), np.arange(2)), axis=-1)
# X_set = np.random.random((2,2,2))

i_split_y, i_split_x = Y_set.ndim, X_set.ndim-1

Y_set_shape, Y_data_shape = Y_set.shape[:i_split_y], Y_set.shape[i_split_y:]
X_set_shape, X_data_shape = X_set.shape[:i_split_x], X_set.shape[i_split_x:]

_temp = list(itertools.product(Y_set.reshape((-1,) + Y_data_shape), X_set.reshape((-1,) + X_data_shape)))
YX_set = np.array(_temp, dtype=[('y', Y_set.dtype, Y_data_shape),
                                ('x', X_set.dtype, X_data_shape)]).reshape(Y_set_shape + X_set_shape)

# YX_set = np.array(list(itertools.product(Y_set.flatten(), X_set.flatten())),
#                   dtype=[('y', Y_set.dtype), ('x', X_set.dtype)]).reshape(Y_set.shape + X_set.shape)

# YX_set = np.array(list(itertools.product(Y_set, X_set)),
#                   dtype=[('y', Y_set.dtype, Y_data_shape), ('x', X_set.dtype, X_data_shape)]).reshape(Y_set_shape + X_set_shape)


# # tt = list(map(tuple, X_set.reshape((-1,) + X_data_shape)))
# tt = [(x,) for x in X_set.reshape((-1,) + X_data_shape)]
# # tt = list(X_set.reshape((-1,) + X_data_shape))
# xx = np.array(tt, dtype=[('x', X_set.dtype, X_data_shape)]).reshape(X_set_shape)    ###




# val = DirichletRE(YX_set.size, np.ones(YX_set.shape)/YX_set.size).rvs()
# prior = DeterministicRE(val)

alpha_0 = 10*YX_set.size
mean = DirichletRE(YX_set.size, np.ones(YX_set.shape) / YX_set.size).rvs()
prior = DirichletRE(alpha_0, mean, rng)

theta = FiniteRE(YX_set, prior.rvs(), rng)


# theta_m_pmf = theta.p.reshape((-1,) + X_set_shape).sum(axis=0)
# theta_m = FiniteRE(X_set, theta_m_pmf)
# theta_m.mean
# theta_m.rvs()
# theta_m.pmf(theta_m.rvs(2))


#%% Classes

# TODO: make special rv classes for supervised learning structured arrays?

class Model(BaseRE):
    def __init__(self, theta_x, theta_y_x, seed=None):
        super().__init__(theta_x, theta_y_x, seed=seed)

    @property
    def theta_x(self):
        return self._theta_x

    @theta_x.setter
    def theta_x(self, theta_x):
        self._update_attr(theta_x=theta_x)

    @property
    def theta_y_x(self):
        return self._theta_y_x

    @theta_y_x.setter
    def theta_y_x(self, theta_y_x):
        self._update_attr(theta_y_x=theta_y_x)

    def _update_attr(self, *args, **kwargs):
        if 'theta_x' in kwargs.keys():
            self._theta_x = kwargs['theta_x']
            self.data_shape_x = self._theta_x.data_shape
        elif len(args) > 0:
            self._theta_x = args[0]
            self.data_shape_x = self._theta_x.data_shape

        if 'theta_y_x' in kwargs.keys():
            self._theta_y_x = kwargs['theta_y_x']
            self.data_shape_y = self._theta_y_x(self._theta_x.rvs()).data_shape
        elif len(args) > 1:
            self._theta_y_x = args[1]
            self.data_shape_y = self._theta_y_x(self._theta_x.rvs()).data_shape

    def _rvs(self, size=(), random_state=None):
        X = np.asarray(self.theta_x.rvs(size, random_state))
        if len(size) == 0:
            Y = self.theta_y_x(X).rvs(size, random_state)
            D = np.array((Y, X), dtype=[('y', Y.dtype, self.data_shape_y), ('x', X.dtype, self.data_shape_x)])
        else:
            Y = np.asarray([self.theta_y_x(x).rvs((), random_state) for x in X.reshape((-1,) + self.theta_x.data_shape)])\
                .reshape(size + self.data_shape_y)
            D = np.array(list(zip(Y.reshape((-1,) + self.data_shape_y), X.reshape((-1,) + self.data_shape_x))),
                         dtype=[('y', Y.dtype, self.data_shape_y), ('x', X.dtype, self.data_shape_x)]).reshape(size)

        return D


theta_m = DirichletRE(4, [.5, .5])
def theta_c(x): return FiniteRE(['a', 'b'], x)

t = Model(theta_m, theta_c)
t.rvs()
t.rvs(4)


class Prior(multi_rv_generic):
    def __init__(self, dist, seed=None):
        super(Prior, self).__init__(seed)
        self.dist = dist    # stats-like RV object


class DatPriorDoe(Prior):
    def __init__(self, loc, scale, seed=None):
        dist = stats.rayleigh(loc, scale)
        super().__init__(dist, seed)

    def random_model(self):
        a, b = self.dist.rvs(size=2)
        theta_m = stats.beta(a, b)
        def theta_c(x): return stats.beta(5*x, 5*(1-x))     # TODO: just combine these in a rv_obj?

        return Model(theta_m, theta_c)




class FiniteSetPrior(Prior):
    def __init__(self, dist, support, seed=None):
        super(FiniteSetPrior, self).__init__(dist, seed)
        self.support = support
        # self.support_y = np.unique(support['y'])
        # self.support_x = np.unique(support['x'])

    def random_model(self):
        theta_pmf = self.dist.rvs()
        # theta_pmf_m = None

        theta = FiniteRE(self.support, theta_pmf, self.random_state)
        # theta_m = FiniteRE(self.support_x, theta_pmf_m, self.random_state)

        return theta

    @classmethod
    def dirichlet_prior(cls, alpha_0, mean, support, seed=None):
        return cls(DirichletRE(alpha_0, mean), support, seed)






#%%

# prior = DeterministicDiscretePrior(val=mean, support=YX_set, seed=rng)
# prior = DirichletDiscretePrior(alpha_0, mean, support=YX_set, seed=rng)
# prior = DatPriorDoe(0, 1, rng)

# model = prior.random_model()
# D = model.rvs((3, 2))


