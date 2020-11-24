"""
Main.
"""

import numpy as np
from matplotlib import pyplot as plt

from thesis.random import elements as rand_elements, models as rand_models
from thesis.bayes import models as bayes_models
from thesis.predictors import (ModelRegressor, BayesRegressor, ModelClassifier, BayesClassifier,
                               plot_loss_eval_compare, plot_predict_stats_compare)

# plt.style.use('seaborn')


#%% Sim

# model = rand_models.NormalRegressor(weights=np.ones(2), basis_y_x=None, cov_y_x=1., model_x=Normal(0, 10), rng=None)
# model = rand_models.ClassConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                  [0, .5], p_y=None)
model = rand_models.DataConditional.from_finite([rand_elements.Finite(['a', 'b'], [p, 1 - p]) for p in (.6, .6)],
                                                [0, .5], p_x=None)

# prior_mean = model
prior_mean = rand_models.DataConditional.from_finite([rand_elements.Finite(['a', 'b'], [p, 1 - p]) for p in (.8, .8)],
                                                     [0, .5], p_x=None)

# Plotting
predictors = [
    # ModelRegressor(model, name=r'$f_{opt}$'),
    # BayesRegressor(bayes_models.NormalRegressor(prior_mean=0 * np.ones(2), prior_cov=0.5 * np.eye(2),
    #                                             basis_y_x=None, cov_y_x=1., model_x=Normal(0, 10)), name='Norm'),
    # BayesRegressor(bayes_models.Dirichlet(prior_mean, 4), name='Dir'),
    BayesClassifier(bayes_models.Dirichlet(prior_mean, 4), name='Dir'),
]

# predictors[0].plot_loss_eval(params={'weights': np.linspace(0, 2, 20)}, n_train=[0, 1, 2], n_test=10, n_mc=100, verbose=True)
# predictors[1].plot_loss_eval(model=None, params={'prior_cov': np.linspace(0.1, 1, 90, endpoint=False)},
#                              n_train=[10], n_test=10, n_mc=400, verbose=True, ax=None, rng=None)

params = [
    # {},
    # {'prior_cov': [0.1, 1, 10]},
    {'alpha_0': [.4, 4]},
    # {'alpha_0': np.arange(.01, 10, .5)}
]

n_train = np.arange(0, 50, 2)
# n_train = [0, 2, 8]

# plot_predict_stats_compare(predictors, model, params, x=None, n_train=2, n_mc=30, do_std=True, ax=None, rng=None)
plot_loss_eval_compare(predictors, model, params, n_train=n_train, n_test=10, n_mc=500,
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
