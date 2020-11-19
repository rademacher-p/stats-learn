"""
Main.
"""

import itertools

import numpy as np

# from scipy.stats._multivariate import multi_rv_generic
# from scipy._lib._util import check_random_state
# from mpl_toolkits.mplot3d import Axes3D

from thesis.random.elements import Normal
from thesis.random.models import NormalRegressor as NormalRegressorModel
from thesis.bayes.models import NormalRegressor as NormalRegressorBayes
from thesis.predictors import BayesRegressor, ModelRegressor, plot_loss_eval_compare

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
# theta = Finite(supp_yx, theta_pmf, rng)
#
# theta_m_pmf = theta_pmf.reshape((-1,) + supp_shape_x).sum(axis=0)
# theta_m = Finite(supp_x_s['x'], theta_m_pmf)
# theta_m_s = Finite(supp_x_s, theta_m_pmf)


#%% Sim

# TODO: split learning funcs from eval funcs for fixed predictors?

def main():
    model_x = Normal(mean=0., cov=10.)
    x = np.linspace(-3, 3, 100, endpoint=False)

    # model_x = Normal(mean=np.zeros(2), cov=np.eye(2))
    # x1_plot = np.linspace(-3, 3, 101, endpoint=True)
    # x2_plot = np.linspace(-3, 3, 81, endpoint=True)
    # x = np.stack(np.meshgrid(x1_plot, x2_plot), axis=-1)

    # model_x = Beta(a=1, b=1)
    # x = np.linspace(0, 1, 100, endpoint=False)

    model = NormalRegressorModel(weights=np.ones(2), basis_y_x=None, cov_y_x=1., model_x=model_x, rng=None)

    # Plotting
    predictors = [
        ModelRegressor(model, name=r'$f_{opt}$'),
        BayesRegressor(NormalRegressorBayes(prior_mean=0 * np.ones(2), prior_cov=0.5 * np.eye(2), basis_y_x=None, cov_y_x=1., model_x=model_x), name='Norm'),
    ]

    # predictors[0].plot_loss_eval(params={'weights': np.linspace(0, 2, 20)}, n_train=[0, 1, 2], n_test=10, n_mc=100, verbose=True)
    # predictors[1].plot_loss_eval(model=None, params={'prior_cov': np.linspace(0.1, 1, 90, endpoint=False)},
    #                              n_train=[10], n_test=10, n_mc=400, verbose=True, ax=None, rng=None)

    params = [
        {},
        {'prior_cov': [0.1, 1, 10]},
    ]

    # plot_predict_stats_compare(predictors, x, model, params, n_train=2, n_mc=30, do_std=True, ax=None, rng=None)
    plot_loss_eval_compare(predictors, model, params, n_train=np.arange(3), n_test=10, n_mc=100,
                           verbose=False, ax=None, rng=100)

    # single predictor methods
    pr = predictors[1]
    pr.set_params(cov_prior=5)

    # params = None
    # params = {'weights': [m * np.ones(2) for m in [.1, .5, 1]]}
    params = {
        # 'prior_cov': [.1, 1, 10, 11],
        'prior_cov': np.linspace(.1, 10, 32),
        # 'prior_mean': [m * np.ones(2) for m in [.1, .5, 1]],
              }

    # n_train = 2
    n_train = [0, 1, 2]
    # n_train = np.arange(10)

    # pr.plot_predict_stats(x, model=model, params=params, n_train=n_train, n_mc=30, do_std=True, ax=None, rng=None)
    pr.plot_loss_eval(model=model, params=params, n_train=n_train, n_test=10, n_mc=100, verbose=False, ax=None, rng=100)

    pass

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
