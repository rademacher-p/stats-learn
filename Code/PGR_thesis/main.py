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
supp_x = np.linspace(0, 1, 16, endpoint=True)
model_x = rand_elements.FiniteRV(supp_x, p=None)


def weights_to_mean(weights):
    return sum(w * supp_x ** i for i, w in enumerate(weights))


def mean_to_rv(mean):
    return rand_elements.BinomialNormalized(10, mean)
    # return rand_elements.Beta.from_mean(50, mean)


# model = rand_models.NormalLinear(weights=np.ones(2), basis_y_x=None, cov_y_x=1.,
#                                  model_x=rand_elements.Normal(0, 10), rng=None)
# model = rand_models.ClassConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                  ['a', 'b'], p_y=None)
# model = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                 supp_x, p_x=None)
# prior_mean = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.9, .9)],
#                                                      supp_x, p_x=None)

# mean_y_x = weights_to_mean(w_prior)
mean_y_x = weights_to_mean([0, 0, 1])
# mean_y_x = 0.5 + 0.5 * np.sin(2*np.pi * supp_x)
# mean_y_x = 1 / (1 + 4 * supp_x ** 4)
model = rand_models.DataConditional(list(map(mean_to_rv, mean_y_x)), model_x)

# w_prior = np.array([0, 0, 1])
w_prior = np.array([.5, 0])

mean_y_x_dir = weights_to_mean(w_prior)
prior_mean = rand_models.DataConditional(list(map(mean_to_rv, mean_y_x_dir)), model_x)
# model = bayes_models.Dirichlet(prior_mean, alpha_0=4)


# Plotting
predictors = [
    ModelRegressor(model, name=r'$f_{opt}$'),
    BayesRegressor(bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=10 * np.eye(w_prior.size),
                                             basis_y_x=None, cov_y_x=.1,
                                             model_x=model_x), name='Norm'),
    BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=.1), name='Dir'),
    # BayesRegressor(model),
    # BayesClassifier(bayes_models.Dirichlet(prior_mean, alpha_0=40), name='Dir'),
]

# predictors[0].plot_loss_eval(params={'weights': np.linspace(0, 2, 20)}, n_train=[0, 1, 2], n_test=10, n_mc=100, verbose=True)
# predictors[1].plot_loss_eval(model=None, params={'prior_cov': np.linspace(0.1, 1, 90, endpoint=False)},
#                              n_train=[10], n_test=10, n_mc=400, verbose=True, ax=None, rng=None)

params = [
    {},
    # {},
    # {'prior_cov': [10, 0.05]},
    {'prior_cov': [10]},
    # {'prior_mean.p_x': [[.7,.3], [.4,.6]]},
    # {},
    # {'alpha_0': [.1, 50]},
    {'alpha_0': [.1]},
    # {'alpha_0': np.arange(.01, 100, 5)}
]

# n_train = np.arange(0, 200, 5)
# n_train = [0, 100, 200]
n_train = 100

# plot_loss_eval_compare(predictors, model, params, n_train=n_train, n_test=10, n_mc=100,
#                        verbose=True, ax=None, rng=None)
plot_predict_stats_compare(predictors, model, params, x=None, n_train=n_train, n_mc=1000, do_std=True,
                           verbose=True, ax=None, rng=None)

plt.show()

# single predictor methods
pr = predictors[0]
pr.set_params(cov_prior=5)

# params = None
# params = {'weights': [m * np.ones(2) for m in [.1, .5, 1]]}
# params = {
#     # 'prior_cov': [.1, 1, 10, 11],
#     'prior_cov': np.linspace(.1, 10, 32),
#     # 'prior_mean': [m * np.ones(2) for m in [.1, .5, 1]],
#           }
params = {'alpha_0': np.linspace(0, 10, 20, endpoint=False)}

# n_train = 2
n_train = [0, 2, 4, 8]
# n_train = np.arange(10)

pr.plot_predict_stats(model=model, params=params, x=None, n_train=n_train, n_mc=30, do_std=True, ax=None, rng=None)
# pr.plot_loss_eval(model=model, params=params, n_train=n_train, n_test=10, n_mc=100, verbose=False, ax=None, rng=100)
