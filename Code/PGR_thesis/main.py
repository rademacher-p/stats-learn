"""
Main.
"""

import itertools
from collections.abc import Sequence

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

# from scipy.stats._multivariate import multi_rv_generic
# from scipy._lib._util import check_random_state
# from mpl_toolkits.mplot3d import Axes3D

from util.generic import vectorize_first_arg
from random_elements import Normal, Beta
from models import NormalRegressor as NormalRegressorModel
from bayes import NormalRegressor as NormalRegressorBayes
from decision_functions.learn_funcs import (BayesPredictor, BayesClassifier, BayesRegressor,
                                            ModelPredictor, ModelClassifier, ModelRegressor)

# plt.style.use('seaborn')


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

supp_y = np.array(['a', 'b'])
# supp_y = np.arange(2) / 2
supp_x = np.arange(2) / 2
# supp_x = np.arange(6).reshape(3, 2)
# supp_x = np.stack(np.meshgrid(np.arange(2), np.arange(3)), axis=-1)

i_split_y, i_split_x = supp_y.ndim, supp_x.ndim - 0

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


# alpha_0 = 10 * supp_yx.size
# mean = Dirichlet(supp_yx.size, np.ones(supp_yx.shape) / supp_yx.size).rvs()
# prior = Dirichlet(alpha_0, mean, rng)
#
# theta_pmf = prior.rvs()
# theta = FiniteRE(supp_yx, theta_pmf, rng)
#
# theta_m_pmf = theta_pmf.reshape((-1,) + supp_shape_x).sum(axis=0)
# theta_m = FiniteRE(supp_x_s['x'], theta_m_pmf)
# theta_m_s = FiniteRE(supp_x_s, theta_m_pmf)


#%% Sim

# TODO: want RNG seeding for identical fit/eval data
# TODO: split learning funcs from eval funcs for fixed predictors?
# TODO: need MC results for fitting

def predictor_compare(predictors, model, n_train, n_test, rng=None):
    d = model.rvs(n_train + n_test, rng=rng)  # generate train/test data
    d_train, d_test = np.split(d, [n_train])

    losses = np.empty(len(predictors))
    for idx, predictor in enumerate(predictors):
        predictor.fit(d_train)
        losses[idx] = predictor.evaluate(d_test)
    return losses


def predictor_compare_mc(predictors, model, n_train=0, n_test=1, n_mc=1, rng=None):
    model.rng = rng
    loss_mc = np.empty((n_mc, len(predictors)))
    for i_mc in range(n_mc):
        loss_mc[i_mc] = predictor_compare(predictors, model, n_train, n_test, rng=None)
    return loss_mc.mean(0)


def predictor_compare_mc_bayes(predictors, bayes_model, n_train=0, n_test=1, n_mc=1, rng=None):
    bayes_model.rng = rng
    loss_mc = np.empty((n_mc, len(predictors)))
    for i_mc in range(n_mc):
        model = bayes_model.random_model()
        loss_mc[i_mc] = predictor_compare(predictors, model, n_train, n_test, rng=None)
    return loss_mc.mean(0)


# def predictor_eval_2(predictors, model, rng_fit=None, rng_eval=None):
#     losses = []
#     for predictor in predictors:
#         predictor.fit_from_model(model, n_train=5, rng=rng_fit)
#         loss = predictor.evaluate_from_model(model, n_test=10, n_mc=15, rng=rng_eval)
#         losses.append(loss)
#
#     return losses


def main():
    model_x = Normal(mean=0., cov=1.)
    x_plt = np.linspace(-3, 3, 100, endpoint=False)

    # model_x = Normal(mean=np.zeros(2), cov=np.eye(2))
    # x1_plot = np.linspace(-3, 3, 101, endpoint=True)
    # x2_plot = np.linspace(-3, 3, 81, endpoint=True)
    # x_plt = np.stack(np.meshgrid(x1_plot, x2_plot), axis=-1)

    # model_x = Beta(a=1, b=1)
    # x_plt = np.linspace(0, 1, 100, endpoint=False)

    model = NormalRegressorModel(model_x=model_x, basis_y_x=None,  # (lambda x: 1., lambda x: x)
                                 weights=np.ones(2), cov_y_x_single=1., rng=None)

    bayes_models = {r'$C_{\theta} = $' + str(_cov): NormalRegressorBayes(model_x=model_x, basis_y_x=None, cov_y_x=1.,
                                                                         mean_prior=np.zeros(2),
                                                                         cov_prior=_cov*np.eye(2))
                    for _cov in [0.1, 10]}

    predictors = [
        ModelRegressor(model, name=r'$f_{opt}$'),
        *(BayesRegressor(bayes_model, name=name) for name, bayes_model in bayes_models.items()),
    ]

    # Risk sim
    losses = predictor_compare_mc(predictors, model, n_train=10, n_test=1, n_mc=20, rng=None)
    print(losses)

    # losses = predictor_compare_mc_bayes(predictors, bayes_models['learn: 0.1'],
    #                                     n_train=10, n_test=1, n_mc=3, rng=None)
    # print(losses)

    # Plotting

    subplot_kw = {'projection': '3d'} if model_x.shape == (2,) else {}
    _, ax = plt.subplots(subplot_kw=subplot_kw)
    n_train = 10
    ModelPredictor.plot_compare_stats(predictors, x_plt, model, n_train, n_mc=50, do_std=True, ax=ax, rng=None)
    ax.legend()
    ax.grid(True)
    ax.set_title(f'N = {n_train}')

    _, axn = plt.subplots()
    pr = predictors[1]
    pr.fit()
    n_c = 0
    for n_train in [0, 5, 10, 40]:
        pr.fit_from_model(model, n_train, warm_start=True)
        n_c += n_train
        pr.plot_predict(x_plt, ax=axn, label=f"N = {n_c}")
    axn.grid(True)
    axn.legend()
    axn.set_title(f"{pr.name}")

    subplot_kw = {'projection': '3d'} if model_x.shape == (2,) else {}
    _, ax = plt.subplots(subplot_kw=subplot_kw)
    pr = predictors[1]
    n_train = [0, 5, 10, 50]
    pr.plot_predict_stats(x_plt, model, n_train, n_mc=50, do_std=True, ax=ax, rng=None)
    ax.grid(True)
    ax.legend()
    ax.set_title(f"{pr.name}")


# def main():
#     alpha_0 = alpha_0_plot = supp_x_s.size * supp_y_s.size
#
#     # mean = np.ones(supp_x_s.shape + supp_y_s.shape) / (supp_x_s.size * supp_y_s.size)
#
#     mean_x = FiniteDomainFunc(supp_x, np.ones(supp_x_s.shape) / supp_x_s.size)
#
#     mean_y_x = FiniteDomainFunc(supp_x, np.full(supp_x_s.shape,
#                                                 FiniteDomainFunc(supp_y, np.ones(supp_y_s.shape) / supp_y_s.size)))
#
#     # bayes_model = DirichletFiniteYcXModelBayes(supp_x_s, supp_y_s, alpha_0, mean,
#     #                                            rng_model=default_rng(6),
#     #                                            rng_prior=default_rng(5))
#     bayes_model = DirichletFiniteYcXModelBayesNew(alpha_0, mean_x, mean_y_x,
#                                                   rng_model=default_rng(6),
#                                                   rng_prior=default_rng(5))
#
#     learner = ModelClassifier(bayes_model)
#
#     loss = learn_eval_mc_bayes(bayes_model, learner, n_train=10, n_test=1, n_mc=5, verbose=False)
#
#     bayes_model = BetaModelBayes()
#     learner = BetaEstimatorTemp(n_x=10)


if __name__ == '__main__':
    main()
