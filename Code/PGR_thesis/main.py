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


#%% Sim

# supp_x = np.array([0, .5])
supp_x = np.linspace(0, 1, 16, endpoint=True)

norm_mean = [1, -1]

# model = rand_models.NormalLinear(weights=np.ones(2), basis_y_x=None, cov_y_x=1.,
#                                  model_x=rand_elements.Normal(0, 10), rng=None)
# model = rand_models.ClassConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                  ['a', 'b'], p_y=None)
# model = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                 supp_x, p_x=None)
model = rand_models.DataConditional.from_finite([rand_elements.BinomialNormalized(10, p) for p in supp_x ** 2],
                                                supp_x, p_x=None)

# prior_mean = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.9, .9)],
#                                                      supp_x, p_x=None)
dir_p_y_x = sum(w * supp_x ** i for i, w in enumerate(norm_mean))
prior_mean = rand_models.DataConditional.from_finite([rand_elements.BinomialNormalized(10, p) for p in dir_p_y_x],
                                                     supp_x, p_x=None)
# model = bayes_models.Dirichlet(prior_mean, alpha_0=4)


# Plotting
predictors = [
    ModelRegressor(model, name=r'$f_{opt}$'),
    BayesRegressor(bayes_models.NormalLinear(prior_mean=norm_mean, prior_cov=0.5 * np.eye(2),
                                             basis_y_x=None, cov_y_x=1.,
                                             model_x=rand_elements.Uniform(0, 1)), name='Norm'),
    BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=4), name='Dir'),
    # BayesRegressor(model),
    # BayesClassifier(bayes_models.Dirichlet(prior_mean, alpha_0=40), name='Dir'),
]

# predictors[0].plot_loss_eval(params={'weights': np.linspace(0, 2, 20)}, n_train=[0, 1, 2], n_test=10, n_mc=100, verbose=True)
# predictors[1].plot_loss_eval(model=None, params={'prior_cov': np.linspace(0.1, 1, 90, endpoint=False)},
#                              n_train=[10], n_test=10, n_mc=400, verbose=True, ax=None, rng=None)

params = [
    {},
    # {},
    # {'cov_y_x': [.1, 1]}
    {'prior_cov': [10, 0.1]},
    # {'prior_mean.p_x': [[.7,.3], [.4,.6]]},
    # {},
    {'alpha_0': [.1, 10]},
    # {'alpha_0': np.arange(.01, 10, .5)}
]

n_train = np.arange(0, 50, 5)
# n_train = [0, 2, 8]
# n_train = 10

# plot_predict_stats_compare(predictors, model, params, x=None, n_train=2, n_mc=30, do_std=True, ax=None, rng=None)
plot_loss_eval_compare(predictors, model, params, n_train=n_train, n_test=10, n_mc=1000,
                       verbose=True, ax=None, rng=None)

# single predictor methods
pr = predictors[1]
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

# pr.plot_predict_stats(model=model, params=params, x=None, n_train=n_train, n_mc=30, do_std=True, ax=None, rng=None)
# pr.plot_loss_eval(model=model, params=params, n_train=n_train, n_test=10, n_mc=100, verbose=False, ax=None, rng=100)
