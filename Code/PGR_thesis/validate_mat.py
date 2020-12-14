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

supp_x = np.array([0, .5])
model_x = rand_elements.FiniteRV(supp_x, p=None)

# model = rand_models.ClassConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                  ['a', 'b'], p_y=None)
model = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)], model_x)
# model = bayes_models.Dirichlet(model, alpha_0=3)

prior_mean = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.9, .9)], model_x)


# Plotting
predictors = [
    # ModelRegressor(model, name=r'$f_{opt}$'),
    BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=.1), name='Dir'),
    # BayesClassifier(bayes_models.Dirichlet(prior_mean, alpha_0=40), name='Dir'),
]

# predictors[0].plot_loss_eval(params={'weights': np.linspace(0, 2, 20)}, n_train=[0, 1, 2], n_test=10, n_mc=100, verbose=True)
# predictors[1].plot_loss_eval(model=None, params={'prior_cov': np.linspace(0.1, 1, 90, endpoint=False)},
#                              n_train=[10], n_test=10, n_mc=400, verbose=True, ax=None, rng=None)

params = [
    # {},
    # {},
    # {'alpha_0': [2, 16]},
    # {'alpha_0': [50]},
    {'alpha_0': .01 + np.arange(0, 10, .5)}
    # {'prior_mean.p_x': [[.7,.3], [.4,.6]]},
]

# n_train = np.arange(0, 50, 5)
n_train = [0, 2, 4, 8]
# n_train = 10

plot_loss_eval_compare(predictors, model, params, n_train=n_train, n_test=10, n_mc=5000,
                       verbose=True, ax=None, rng=None)
# plot_predict_stats_compare(predictors, model, params, x=None, n_train=n_train, n_mc=300, do_std=True,
#                            verbose=True, ax=None, rng=None)

plt.show()
