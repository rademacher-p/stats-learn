"""
Main.
"""

import datetime
from copy import deepcopy
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from thesis.bayes import models as bayes_models
from thesis.predictors import Bayes as BayesPredictor
from thesis.predictors import BayesClassifier, BayesRegressor
from thesis.predictors import Model as ModelPredictor
from thesis.predictors import (
    ModelClassifier,
    ModelRegressor,
    loss_eval_compare,
    plot_loss_eval_compare,
    plot_predict_stats_compare,
    predict_stats_compare,
)
from thesis.random import elements as rand_elements
from thesis.random import models as rand_models

# plt.style.use('seaborn')


# def dir_figs(n_train, alpha_0, p_y=(.5, .5), n_mc=1000, do_bayes=False, alpha_0_true=3):
#
#     model_x = rand_elements.FiniteRV([0, .5], p=None)
#
#     model = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in p_y], model_x)
#
#     if do_bayes:
#         model = bayes_models.Dirichlet(model, alpha_0_true)
#
#     if isinstance(alpha_0, Iterable):
#         alpha_0_init = alpha_0[0]
#         dir_params = {'alpha_0': alpha_0}.copy()
#         if isinstance(n_train, Iterable):
#             if len(n_train) >= len(alpha_0):
#                 plot_type = 'n'
#         else:
#             plot_type = 'a'
#     else:
#         alpha_0_init = alpha_0
#         dir_params = None
#         plot_type = 'n'
#         if not isinstance(n_train, Iterable):
#             return
#
#     prior_mean = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)], model_x)
#     dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0_init), name='Dir')
#
#     if plot_type == 'a':
#         dir_predictor.plot_loss_eval(model, params=dir_params, n_train=n_train, n_test=1, n_mc=n_mc, verbose=True,
#                                      rng=None)
#     elif plot_type == 'n':
#         predictors = [
#             ModelRegressor(model, name=r'$f_{opt}$'),
#             dir_predictor,
#         ]
#
#         params = [None, dir_params]
#
#         plot_loss_eval_compare(predictors, model, params, n_train=n_train, n_test=1, n_mc=n_mc,
#                                verbose=True, ax=None, rng=None)
#
#     plt.show()
#     plt.savefig(f'images/{datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")}')
#     plt.close()
#
#
# def main():
#     args = [
#         # (np.arange(0, 50, 1), [2, 16], (.5, .5)),
#         (np.arange(0, 50, 1), [2, 16], (.9, .9)),
#         ([0, 2, 8], .001 + np.arange(0, 10, .1), (.5, .5)),
#         ([0, 2, 8], .001 + np.arange(0, 20, .1), (.9, .9)),
#     ]
#
#     for args_i in args:
#         dir_figs(*args_i, n_mc=50000)
#
#
# if __name__ == '__main__':
#     main()


#%% Sim

# True model
model_x = rand_elements.FiniteRV([0, 0.5], p=None)

model = rand_models.DataConditional(
    [rand_elements.Finite([0, 0.5], [p, 1 - p]) for p in (0.5, 0.5)], model_x
)

# model = bayes_models.Dirichlet(model, alpha_0=3)


# Optimal learner
if isinstance(model, rand_models.Base):
    opt_predictor = ModelRegressor(model, name=r"$f_{\Theta}$")
elif isinstance(model, bayes_models.Base):
    opt_predictor = BayesRegressor(model, name=r"$f^*$")
else:
    raise TypeError


# Dirichlet learner
prior_mean = rand_models.DataConditional(
    [rand_elements.Finite([0, 0.5], [p, 1 - p]) for p in (0.5, 0.5)], model_x
)
dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=3), name="Dir")

# dir_params = None
dir_params = {"alpha_0": [2, 16]}
# dir_params = {'alpha_0': .001 + np.arange(2, 4, .2)}

# print(dir_predictor.loss_eval(model, dir_params, n_train, n_test=1, n_mc=20000, verbose=True, rng=None))
# dir_predictor.plot_loss_eval(model, dir_params, n_train, n_test=1, n_mc=1000, verbose=True, rng=None)


# Plotting

# n_train = 2
# n_train = [2, 4]
n_train = np.arange(0, 11, 1)


predictors = [
    opt_predictor,
    dir_predictor,
]

params = [None, dir_params]


plot_loss_eval_compare(
    predictors,
    model,
    params,
    n_train=n_train,
    n_test=1,
    n_mc=4000,
    verbose=True,
    ax=None,
    rng=None,
)
# plot_predict_stats_compare(predictors, model, params, x=None, n_train=n_train, n_mc=300, do_std=True,
#                            verbose=True, ax=None, rng=None)


if isinstance(model, rand_models.Base):
    risk_an = opt_predictor.risk_min()
    print(f"Min risk = {risk_an}")
elif isinstance(model, bayes_models.Base):
    risk_an = opt_predictor.bayes_risk_min(n_train)
    print(f"Min Bayes risk = {risk_an}")
else:
    raise TypeError

print("Done")
