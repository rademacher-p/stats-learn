"""
Sim main.

blah blah
"""


import numpy as np
from numpy.random import default_rng
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow_probability import distributions as tfd

from simplex_grid import simplex_grid

rng = default_rng()

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
# Y_set = np.array(['a'])
X_set = np.arange(1)

YX_set = np.array([(y, x) for y in Y_set for x in X_set],
                  dtype=[('y', Y_set.dtype), ('x', X_set.dtype)]).reshape(Y_set.shape + X_set.shape)


# alpha = 5*np.ones(Y_set.shape + X_set.shape)
alpha = rng.uniform(1, 10, Y_set.shape + X_set.shape)

t_plt = simplex_grid(10, alpha.size).reshape((-1,) + alpha.shape)


# prior = stats.dirichlet(alpha.flatten())
# prior_plt = prior.pdf(t_plt.reshape((-1, alpha.size)).T)
# theta_pmf = prior.rvs().reshape(alpha.shape)

prior = tfd.Dirichlet(alpha.flatten())
# prior = tfd.Deterministic(rng.choice(t_plt).flatten())    # TODO: error, need multivariate
prior_plt = prior.prob(t_plt.reshape((-1, alpha.size)))
theta_pmf = prior.sample().numpy().reshape(alpha.shape)



if alpha.shape == (3, 1):
    _, ax = plt.subplots(num='prior', clear=True, subplot_kw={'projection': '3d'})
    sc = ax.scatter(t_plt[:, 0], t_plt[:, 1], t_plt[:, 2], s=15, c=prior_plt)
    ax.view_init(35, 45)
    plt.colorbar(sc)


# def prior():
#     theta = stats.dirichlet(alpha.flatten()).rvs().reshape(alpha.shape) # TODO: persistence?
#     return stats.rv_discrete(name='theta', values=(vals, theta.flatten()))


q = rng.choice(YX_set.flatten(), p=theta_pmf.flatten())

z = stats.rv_discrete(name='z', values=(['a','b','c'], [.2,.5,.3]))     # cant handle non-integral values...
z = tfd.Categorical(theta_pmf.flatten())    # only returns integers...



# vals = YX_set
vals = np.arange(alpha.size).reshape(alpha.shape)
# vals = list('asdfer')
theta = stats.rv_discrete(name='theta', values=(vals, theta_pmf))

plt.figure(num='theta_pmf', clear=True)
plt.stem(vals.flatten(), theta.pmf(vals).flatten(), use_line_collection=True)


