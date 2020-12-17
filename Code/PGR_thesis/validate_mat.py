"""
Main.
"""

from copy import deepcopy

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

model = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)], model_x)
model = bayes_models.Dirichlet(model, alpha_0=1)

prior_mean = deepcopy(model.prior_mean)
# prior_mean = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)], model_x)


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
    {'alpha_0': .01 + np.arange(0, 5, .1)}
    # {'prior_mean.p_x': [[.7,.3], [.4,.6]]},
]

# n_train = np.arange(0, 11, 1)
# n_train = [0, 2, 4, 8]
n_train = 2

loss = predictors[0].loss_eval(model, params=None, n_train=n_train, n_test=1, n_mc=100000, verbose=True, rng=None)
print(loss)
# predictors[0].plot_loss_eval(model, params=params[0], n_train=n_train, n_test=1, n_mc=20000, verbose=True, rng=None)


# plot_loss_eval_compare(predictors, model, params, n_train=n_train, n_test=1, n_mc=4000,
#                        verbose=True, ax=None, rng=None)
# plot_predict_stats_compare(predictors, model, params, x=None, n_train=n_train, n_mc=300, do_std=True,
#                            verbose=True, ax=None, rng=None)


bayes_risk = 0.
for x in prior_mean.space['x'].values:
    alpha_m = model.prior_mean.model_x.pf(x)
    weight = (alpha_m + 1 / (model.alpha_0 + n_train)) / (alpha_m + 1 / model.alpha_0)
    bayes_risk += alpha_m * model.prior_mean.model_y_x(x).cov * weight

print(bayes_risk)

qq = 1
