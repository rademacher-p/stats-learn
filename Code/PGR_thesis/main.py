"""
Main.
"""

import numpy as np
from matplotlib import pyplot as plt

from thesis.random.elements import Normal
from thesis.random.models import NormalRegressor as NormalRegressorModel
from thesis.bayes.models import NormalRegressor as NormalRegressorBayes
from thesis.predictors import BayesRegressor, ModelRegressor, plot_loss_eval_compare, plot_predict_stats_compare

# plt.style.use('seaborn')


#%% Sim

def main():
    model_x = Normal(mean=0., cov=10.)
    # x = np.linspace(-3, 3, 100, endpoint=False)

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

    plot_predict_stats_compare(predictors, model, params, x=None, n_train=2, n_mc=30, do_std=True, ax=None, rng=None)
    # plot_loss_eval_compare(predictors, model, params, n_train=np.arange(3), n_test=10, n_mc=100,
    #                        verbose=False, ax=None, rng=100)

    # single predictor methods
    pr = predictors[1]
    pr.set_params(cov_prior=5)

    params = None
    # params = {'weights': [m * np.ones(2) for m in [.1, .5, 1]]}
    # params = {
    #     # 'prior_cov': [.1, 1, 10, 11],
    #     'prior_cov': np.linspace(.1, 10, 32),
    #     # 'prior_mean': [m * np.ones(2) for m in [.1, .5, 1]],
    #           }

    # n_train = 2
    n_train = [0, 1, 2]
    # n_train = np.arange(10)

    pr.plot_predict_stats(model=model, params=params, x=None, n_train=n_train, n_mc=30, do_std=True, ax=None, rng=None)
    # pr.plot_loss_eval(model=model, params=params, n_train=n_train, n_test=10, n_mc=100, verbose=False, ax=None, rng=100)

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
