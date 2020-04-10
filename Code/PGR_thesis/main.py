"""
Sim main.

blah blah
"""

import numpy as np
from numpy import random
from scipy import stats
import matplotlib.pyplot as plt

# from tensorflow_probability import distributions as tfd

from util.util import simplex_grid
from rv_obj import deterministic_multi, dirichlet_multi

rng = random.default_rng()

np.seterr(all='raise')


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

Y_set = np.array(['a', 'b', 'c'])
# Y_set = np.array(['a', 'b'])
X_set = np.arange(2)

YX_set = np.array([(y, x) for y in Y_set for x in X_set],
                  dtype=[('y', Y_set.dtype), ('x', X_set.dtype)]).reshape(Y_set.shape + X_set.shape)

n_plt = 20

# mean = dirichlet_multi.rvs(YX_set.size, np.ones(YX_set.shape)/YX_set.size)
# prior = deterministic_multi(mean)
# t_plt = simplex_grid(n_plt, YX_set.shape)

alpha_0 = 5
mean = dirichlet_multi.rvs(YX_set.size, np.ones(YX_set.shape)/YX_set.size)
prior = dirichlet_multi(alpha_0, mean)
t_plt = simplex_grid(n_plt, YX_set.shape, mean < 1 / alpha_0)


prior_plt = prior.pdf(t_plt)
theta_pmf = prior.rvs()

# prior_plt.sum() / (n_plt**(mean.size-1))


# TODO: add plot methods to RV classes
if YX_set.shape == (3, 1):
    _, ax = plt.subplots(num='prior', clear=True, subplot_kw={'projection': '3d'})
    sc = ax.scatter(t_plt[:, 0], t_plt[:, 1], t_plt[:, 2], s=15, c=prior_plt)
    ax.view_init(35, 45)
    plt.colorbar(sc)
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')



q = rng.choice(YX_set.flatten(), p=theta_pmf.flatten())

z = stats.rv_discrete(name='z', values=(['a', 'b', 'c'], [.2, .5, .3]))     # cant handle non-integral values...


# vals = YX_set
vals = np.arange(YX_set.size).reshape(YX_set.shape)
# vals = list('asdfer')
theta = stats.rv_discrete(name='theta', values=(vals, theta_pmf))

plt.figure(num='theta_pmf', clear=True)
plt.stem(vals.flatten(), theta.pmf(vals).flatten(), use_line_collection=True)


