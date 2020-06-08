"""
Main.

:-)
"""

import itertools
import functools
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from scipy import stats
# from scipy.stats._multivariate import multi_rv_generic
# from scipy._lib._util import check_random_state
# from mpl_toolkits.mplot3d import Axes3D

from RE_obj import DeterministicRE, FiniteRE, DirichletRV
from SL_obj import YcXModel
from bayes import BaseBayes, DirichletYcXModelBayes
from loss_functions import loss_01, loss_se
from learn_functions import DirichletClassifier, DirichletEstimator
from util.generic import empirical_pmf

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
#
#
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

# supp_y = np.array(['a', 'b'])
supp_y = np.arange(2) / 2
# supp = np.arange(2) / 2
# supp = np.arange(6).reshape(3, 2)
supp_x = np.stack(np.meshgrid(np.arange(2), np.arange(3)), axis=-1)

i_split_y, i_split_x = supp_y.ndim, supp_x.ndim - 1

supp_shape_y, data_shape_y = supp_y.shape[:i_split_y], supp_y.shape[i_split_y:]
supp_shape_x, data_shape_x = supp_x.shape[:i_split_x], supp_x.shape[i_split_x:]

supp_yx = np.array(list(itertools.product(supp_y.reshape((-1,) + data_shape_y), supp_x.reshape((-1,) + data_shape_x))),
                   dtype=[('y', supp_y.dtype, data_shape_y),
                          ('x', supp_x.dtype, data_shape_x)]).reshape(supp_shape_y + supp_shape_x)

supp_xy = np.array(list(itertools.product(supp_x.reshape((-1,) + data_shape_x), supp_y.reshape((-1,) + data_shape_y))),
                   dtype=[('x', supp_x.dtype, data_shape_x),
                          ('y', supp_y.dtype, data_shape_y)]).reshape(supp_shape_x + supp_shape_y)

supp_x_s = np.array(list(itertools.product(supp_x.reshape((-1,) + data_shape_x))),
                    dtype=[('x', supp_x.dtype, data_shape_x)]).reshape(supp_shape_x)

supp_y_s = np.array(list(itertools.product(supp_y.reshape((-1,) + data_shape_y))),
                    dtype=[('y', supp_y.dtype, data_shape_y)]).reshape(supp_shape_y)


alpha_0 = 10 * supp_yx.size
mean = DirichletRV(supp_yx.size, np.ones(supp_yx.shape) / supp_yx.size).rvs()
prior = DirichletRV(alpha_0, mean, rng)

theta_pmf = prior.rvs()
theta = FiniteRE(supp_yx, theta_pmf, rng)

theta_m_pmf = theta_pmf.reshape((-1,) + supp_shape_x).sum(axis=0)
theta_m = FiniteRE(supp_x_s['x'], theta_m_pmf)
theta_m_s = FiniteRE(supp_x_s, theta_m_pmf)


#%% Sim

# TODO: vectorize learners and losses!

def learn_sim(prior, learner, n_train=0, n_test=1, n_mc=1, verbose=False):
    loss_mc = np.empty(n_mc)
    for i_mc in range(n_mc):
        if verbose:
            if i_mc % 100 == 0:
                print(f"Iteration {i_mc}/{n_mc}", end='\r')

        theta = prior.random_model()    # randomize model using prior
        d_train, d_test = theta.rvs(n_train), theta.rvs(n_test)     # generate train/test data
        learner.fit(d_train)        # train learner
        loss_mc[i_mc] = learner.evaluate(d_test)        # make decision and assess

    return loss_mc.mean()


