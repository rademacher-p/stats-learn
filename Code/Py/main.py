"""
Sim main.

blah blah
"""


import numpy as np
from numpy.random import default_rng
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rng = default_rng()

# N = 10

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



#%%

Y_set = np.array(['a', 'b'])
X_set = np.arange(3)
# Y_set = tuple(np.array(['a', 'b']))
# X_set = tuple(np.arange(6).reshape(3, 2))

YX_set = np.array([(y, x) for y in Y_set for x in X_set],
                  dtype=[('y', Y_set.dtype), ('x', X_set.dtype)]).reshape(Y_set.shape + X_set.shape)


alpha = 5*np.ones(Y_set.shape + X_set.shape)
# alpha = 5*np.ones((len(Y_set), len(X_set)))
alpha[-1, -1] = 100

prior = stats.dirichlet(alpha.flatten())

plt.figure(num='prior', clear=True)
plt.

# def prior():
#     theta = drc(alpha.flatten()).rvs().reshape(alpha.shape)
#     return stats.rv_discrete(name='theta', values=(vals, theta.flatten()))

theta_pmf = prior.rvs().reshape(alpha.shape)

q = rng.choice(YX_set.flatten(), p=theta_pmf.flatten())

# vals = YX_set
vals = np.arange(1, 7).reshape(2, 3) / 2
# vals = list('asdfer')
theta = stats.rv_discrete(name='theta', values=(vals, theta_pmf))

plt.figure(num='theta_pmf', clear=True)
plt.stem(vals.flatten(), theta.pmf(vals).flatten(), use_line_collection=True)

