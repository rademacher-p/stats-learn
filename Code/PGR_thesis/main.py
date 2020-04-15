"""
Sim main.

:-)
"""

import itertools

import numpy as np
from numpy import random
from scipy import stats
import matplotlib.pyplot as plt

from util.util import simplex_grid
from rv_obj import deterministic_multi, dirichlet_multi, discrete_multi

plt.style.use('seaborn')

rng = random.default_rng()



#%% Continuous sets

theta_m = stats.beta(a=.9, b=.9)


def theta_c(x): return stats.beta(5*x, 5*(1-x))


plt.figure(num='theta', clear=True)

x_plot = np.linspace(0, 1, 101, endpoint=True)
plt.subplot(1, 2, 1)
plt.plot(x_plot, theta_m.pdf(x_plot))
plt.gca().set(title='Marginal Model', xlabel='$x$', ylabel=r'$p_{\theta_m}(x)$')
plt.gca().set_ylim(0)

X = theta_m.rvs()

y_plot = np.linspace(0, 1, 101, endpoint=True)
plt.subplot(1, 2, 2)
plt.plot(y_plot, theta_c(X).pdf(x_plot))
plt.gca().set(title='Conditional Model', xlabel='$y$', ylabel=r'$p_{\theta_c}(y;x)$')
plt.gca().set_ylim(0)

Y = theta_c(X).rvs()

plt.suptitle(f'Model, (X,Y) = ({X:.2f},{Y:.2f})')


#%% Discrete sets

# Y_set = np.array(['a', 'b'])
Y_set = np.arange(2)
X_set = np.arange(6).reshape(3, 2)


YX_set = np.array(list(itertools.product(Y_set.flatten(), X_set.flatten())),
                  dtype=[('y', Y_set.dtype), ('x', X_set.dtype)]).reshape(Y_set.shape + X_set.shape)
# YX_set = np.array(list(itertools.product(Y_set, X_set))).reshape(Y_set.shape + X_set.shape)

n_plt = 5

mean = dirichlet_multi.rvs(YX_set.size, np.ones(YX_set.shape)/YX_set.size)
prior = deterministic_multi(mean)
t_plt = simplex_grid(n_plt, YX_set.shape)

# alpha_0 = YX_set.size
# mean = dirichlet_multi.rvs(YX_set.size, np.ones(YX_set.shape) / YX_set.size)
# prior = dirichlet_multi(alpha_0, mean, rng)
# t_plt = simplex_grid(n_plt, YX_set.shape, mean < 1 / alpha_0)


p_theta_plt = prior.pdf(t_plt)
theta_pmf = prior.rvs()

# prior_plt.sum() / (n_plt**(mean.size-1))


# TODO: add plot methods to RV classes
if YX_set.shape == (3, 1):
    _, ax_prior = plt.subplots(num='prior', clear=True, subplot_kw={'projection': '3d'})
    sc = ax_prior.scatter(t_plt[:, 0], t_plt[:, 1], t_plt[:, 2], s=15, c=p_theta_plt)
    ax_prior.view_init(35, 45)
    plt.colorbar(sc)
    ax_prior.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

# TODO: marginal/conditinal models to alleviate structured array issues?

theta = discrete_multi(YX_set, theta_pmf, rng)

theta.mean  # TODO: broken, tuple product


_, ax_theta = plt.subplots(num='theta pmf', clear=True, subplot_kw={'projection': '3d'})
# ax_theta.scatter(YX_set['x'], YX_set['y'], theta_pmf, c=theta_pmf)
ax_theta.bar3d(YX_set['x'].flatten(), YX_set['y'].flatten(), 0, 1, 1, theta_pmf.flatten(), shade=True)
ax_theta.set(xlabel='$x$', ylabel='$y$')

# plt.figure(num='theta_pmf', clear=True)
# plt.stem(theta.support.flatten(), theta.pmf.flatten(), use_line_collection=True)




