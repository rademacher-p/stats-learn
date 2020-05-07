"""
Sim main.

:-)
"""

import itertools

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# from scipy import stats
# from scipy.stats._multivariate import multi_rv_generic
# from scipy._lib._util import check_random_state
# from mpl_toolkits.mplot3d import Axes3D

from RE_obj import DeterministicRE, FiniteRE, DirichletRV
from SL_obj import YcXModel
from bayes import bayes_re
from loss_functions import loss_01, loss_se
from util.util import empirical_pmf

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


# theta_m = stats.multivariate_normal(mean=[0, 0])
# def theta_c(x): return stats.multivariate_normal(mean=x)
#
#
# _, ax_theta_m = plt.subplots(num='theta_m', clear=True, subplot_kw={'projection': '3d'})
#
# x1_plot = np.linspace(-5, 5, 101, endpoint=True)
# x2_plot = np.linspace(-5, 5, 51, endpoint=True)
# X_plot = np.stack(np.meshgrid(x1_plot, x2_plot), axis=-1)
#
# ax_theta_m.plot_wireframe(X_plot[..., 0], X_plot[..., 1], theta_m.pdf(X_plot))
# # ax_prior[0].plot_wireframe(X_plot[0], X_plot[1], theta_m.pdf(t))
# plt.gca().set(title='Marginal Model', xlabel='$x$', ylabel=r'$p_{\theta_m}(x)$')
#
# X = theta_m.rvs()
#
# _, ax_theta_c = plt.subplots(num='theta_c', clear=True, subplot_kw={'projection': '3d'})
#
# y1_plot = np.linspace(-5, 5, 101, endpoint=True)
# y2_plot = np.linspace(-5, 5, 51, endpoint=True)
# Y_plot = np.stack(np.meshgrid(y1_plot, y2_plot), axis=-1)
#
# ax_theta_c.plot_wireframe(Y_plot[..., 0], Y_plot[..., 1], theta_c(X).pdf(Y_plot))
# plt.gca().set(title='Conditional Model', xlabel='$y$', ylabel=r'$p_{\theta_c}(y;x)$')
#
# Y = theta_c(X).rvs()


#%% Discrete sets

Y_set = np.array(['a', 'b'])
# X_set = np.arange(1)
# X_set = np.arange(6).reshape(3, 2)
X_set = np.stack(np.meshgrid(np.arange(2), np.arange(3)), axis=-1)

i_split_y, i_split_x = Y_set.ndim, X_set.ndim-1

Y_set_shape, Y_data_shape = Y_set.shape[:i_split_y], Y_set.shape[i_split_y:]
X_set_shape, X_data_shape = X_set.shape[:i_split_x], X_set.shape[i_split_x:]

YX_set = np.array(list(itertools.product(Y_set.reshape((-1,) + Y_data_shape), X_set.reshape((-1,) + X_data_shape))),
                  dtype=[('y', Y_set.dtype, Y_data_shape),
                         ('x', X_set.dtype, X_data_shape)]).reshape(Y_set_shape + X_set_shape)

# alpha_0 = 10*YX_set.size
# mean = DirichletRV(YX_set.size, np.ones(YX_set.shape) / YX_set.size).rvs()
# prior = DirichletRV(alpha_0, mean, rng)
#
# theta_pmf = prior.rvs()
# theta = FiniteRE(YX_set, theta_pmf, rng)
#
# X_set_s = np.array([(x,) for x in X_set.reshape((-1,) + X_data_shape)],
#                    dtype=[('x', X_set.dtype, X_data_shape)]).reshape(X_set_shape)
#
# theta_m_pmf = theta_pmf.reshape((-1,) + X_set_shape).sum(axis=0)
# theta_m = FiniteRE(X_set_s['x'], theta_m_pmf)






model_cls = YcXModel        # TODO: any constructor? use factory method on YcX?
# model_kwargs = {'model_x': None, 'model_y_x': None}

alpha_0 = 10
mean = np.ones(YX_set.shape) / YX_set.size
prior = DirichletRV(alpha_0, mean)


def rand_kwargs(self):
    pmf = self.rvs()

    pmf_x = pmf.reshape((-1,) + X_set_shape).sum(axis=0)
    model_x = FiniteRE(X_set, pmf_x)

    def model_y_x(x):
        _temp = pmf.reshape(Y_set_shape + (-1,))[..., np.all(x.flatten() == model_x._supp_flat, axis=-1)]
        pmf_y_x = _temp.reshape(Y_set_shape) / _temp.sum()
        return FiniteRE(Y_set, pmf_y_x)

    return {'model_x': model_x, 'model_y_x': model_y_x}


theta = bayes_re(model_cls, prior, rand_kwargs)
theta.random_model()


#%% Sim

# TODO: vectorize learners and losses!

N_mc = 100
N_train, N_test = 10, 10

loss_fcn = loss_01

loss_mc = np.empty(N_mc)
for i_mc in range(N_mc):
    theta.random_model()    # randomize model using prior

    # Generate data
    D_train, D_test = theta.rvs(N_train), theta.rvs(N_test)


    # Train


    # Evaluate
    # loss_mc[i_mc] = loss_fcn(decision_fcn(x), y)


