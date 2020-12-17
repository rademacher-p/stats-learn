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

model_x = rand_elements.FiniteRV([0, .5], p=None)

model = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)], model_x)

model = bayes_models.Dirichlet(model, alpha_0=3)


prior_mean = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)], model_x)
dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=3), name='Dir')


# Plotting

# n_train = 2
n_train = [0, 2, 4]
# n_train = np.arange(0, 11, 1)

# dir_params = None
dir_params = {'alpha_0': .01 + np.arange(0, 10, .2)}


#
# loss = dir_predictor.loss_eval(model, params=dir_params, n_train=n_train, n_test=1, n_mc=20000, verbose=True, rng=None)
# print(loss)

dir_predictor.plot_loss_eval(model, params=dir_params, n_train=n_train, n_test=1, n_mc=3000, verbose=True, rng=None)

bayes_risk = 0.
for x in model.space['x'].values:
    alpha_m = model.prior_mean.model_x.pf(x)
    weight = (alpha_m + 1 / (model.alpha_0 + n_train)) / (alpha_m + 1 / model.alpha_0)
    bayes_risk += alpha_m * model.prior_mean.model_y_x(x).cov * weight

print(bayes_risk)


#
predictors = [
    ModelRegressor(model, name=r'$f_{opt}$'),
    dir_predictor,
]

params = [None, dir_params]

# plot_loss_eval_compare(predictors, model, params, n_train=n_train, n_test=1, n_mc=4000,
#                        verbose=True, ax=None, rng=None)
# plot_predict_stats_compare(predictors, model, params, x=None, n_train=n_train, n_mc=300, do_std=True,
#                            verbose=True, ax=None, rng=None)


qq = 1
