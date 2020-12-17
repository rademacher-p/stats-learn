"""
Main.
"""

import numpy as np
from matplotlib import pyplot as plt

from thesis.random import elements as rand_elements, models as rand_models
from thesis.bayes import models as bayes_models
from thesis.predictors import (ModelRegressor, BayesRegressor, ModelClassifier, BayesClassifier,
                               plot_loss_eval_compare, plot_predict_stats_compare,
                               loss_eval_compare, predict_stats_compare)

# plt.style.use('seaborn')


# %% Sim

# supp_x = np.array([0, .5])
supp_x = np.linspace(0, 1, 11, endpoint=True)

model_x = rand_elements.FiniteRV(supp_x, p=None)


def poly(x, weights):
    return sum(w * x ** i for i, w in enumerate(weights))


def mean_to_rv(mean):
    return rand_elements.BinomialNormalized(supp_x.size, mean)
    # return rand_elements.Beta.from_mean(50, mean)


mean_y_x = poly(supp_x, [.5, 0, 0])
# mean_y_x = poly(supp_x, [.3, 0, .4])
# mean_y_x = 0.5 + 0.5 * np.sin(2*np.pi * supp_x)
# mean_y_x = 1 / (1 + np.exp(10 * supp_x))
# mean_y_x = 1 / (2 + np.sin(2*np.pi * supp_x))
model = rand_models.DataConditional(list(map(mean_to_rv, mean_y_x)), model_x)

# model = rand_models.NormalLinear(weights=np.ones(2), basis_y_x=None, cov_y_x=1.,
#                                  model_x=rand_elements.Normal(0, 10), rng=None)
# model = rand_models.ClassConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                  ['a', 'b'], p_y=None)

model = bayes_models.Dirichlet(model, alpha_0=5)


w_prior = [.5, 0, 0]
# w_prior = [.5, 0, .5]
mean_y_x_dir = poly(supp_x, w_prior)
prior_mean = rand_models.DataConditional(list(map(mean_to_rv, mean_y_x_dir)), model_x)

dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=1), name='Dir')


# Plotting

n_train = 2
# n_train = [0, 10, 20]
# n_train = np.arange(0, 200, 5)

# dir_params = None
dir_params = {'alpha_0': .001 + np.arange(0, 10, .2)}


#
# loss = dir_predictor.loss_eval(model, params=None, n_train=n_train, n_test=1, n_mc=20000, verbose=True, rng=None)
# print(loss)

dir_predictor.plot_loss_eval(model, params=dir_params, n_train=n_train, n_test=1, n_mc=2000, verbose=True, rng=None)

# bayes_risk = 0.
# for x in model.space['x'].values:
#     alpha_m = model.prior_mean.model_x.pf(x)
#     weight = (alpha_m + 1 / (model.alpha_0 + n_train)) / (alpha_m + 1 / model.alpha_0)
#     bayes_risk += alpha_m * model.prior_mean.model_y_x(x).cov * weight
#
# print(bayes_risk)


#
predictors = [
    ModelRegressor(model, name=r'$f_{opt}$'),
    # BayesRegressor(bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=10 * np.eye(len(w_prior)),
    #                                          basis_y_x=None, cov_y_x=.1,
    #                                          model_x=model_x), name='Norm'),
    dir_predictor,
]

params = [
    None,
    # None,
    # {'prior_cov': [10, 0.05]},
    # {'prior_cov': [10]},
    dir_params,
]


# plot_loss_eval_compare(predictors, model, params, n_train=n_train, n_test=1, n_mc=2000,
#                        verbose=True, ax=None, rng=None)
# plot_predict_stats_compare(predictors, model, params, x=None, n_train=n_train, n_mc=300, do_std=True,
#                            verbose=True, ax=None, rng=None)
