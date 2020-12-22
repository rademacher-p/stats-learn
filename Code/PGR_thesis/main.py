"""
Main.
"""

import numpy as np
from matplotlib import pyplot as plt

from thesis.random import elements as rand_elements, models as rand_models
from thesis.bayes import models as bayes_models
from thesis.predictors import (ModelRegressor, BayesRegressor, ModelClassifier, BayesClassifier,
                               plot_risk_eval_sim_compare, plot_predict_stats_compare,
                               plot_risk_eval_comp_compare,
                               risk_eval_sim_compare, predict_stats_compare)

# plt.style.use('seaborn')


#%% Sim

# def poly(x, weights):
#     return sum(w * x ** i for i, w in enumerate(weights))
#
#
# def mean_to_rv(mean):
#     return rand_elements.EmpiricalScalar(supp_x.size, mean)
#     # return rand_elements.Beta.from_mean(mean, 1000)


# def poly_mean_to_models(model_x_, weights):
#     x = model_x_.space.values_flat
#     mean_y_x = sum(w * x ** i for i, w in enumerate(weights))
#     return [rand_elements.EmpiricalScalar(model_x_.space.set_size, mean) for mean in mean_y_x]


# def poly_mean_to_models(model_x_, weights):
#     return func_mean_to_models(model_x_, lambda x: sum(w * x ** i for i, w in enumerate(weights)))
#
#
# def func_mean_to_models(model_x_, func):
#     return [rand_elements.EmpiricalScalar(model_x_.space.set_size, func(x)) for x in model_x_.space.values_flat]


def poly_mean_to_models(x, weights):
    return func_mean_to_models(x, lambda x_: sum(w * x_ ** i for i, w in enumerate(weights)))


def func_mean_to_models(n, func):
    return [rand_elements.EmpiricalScalar(len(x), func(x_i)) for x_i in x]


# True model

# supp_x = np.array([0, .5])
# supp_x = np.linspace(0, 1, 11, endpoint=True)
# model_x = rand_elements.FiniteRV(supp_x, p=None)


# model = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                 supp_x=[0, .5], p_x=None)


w_model = [.5, 0, 0]
# w_model = [0, 0, 1]

# model_x = rand_elements.FiniteRV(np.linspace(0, 1, 11, endpoint=True), p=None)
# model = rand_models.DataConditional(poly_mean_to_models(model_x, w_model), model_x)
# model = rand_models.DataConditional(func_mean_to_models(model_x, lambda x: 1 / (2 + np.sin(2*np.pi * x))), model_x)

# supp_x = np.linspace(0, 1, 11, endpoint=True)
# model = rand_models.DataConditional.from_finite(poly_mean_to_models(supp_x, w_model),
#                                                 supp_x=supp_x, p_x=None)
# model = rand_models.DataConditional.from_finite(func_mean_to_models(supp_x, lambda x: 1 / (2 + np.sin(2*np.pi * x))),
#                                                 supp_x=supp_x, p_x=None)


model = rand_models.BetaLinear(weights=w_model, basis_y_x=None, alpha_y_x=1000, model_x=rand_elements.Beta())


# model = rand_models.DataConditional(list(map(mean_to_rv, poly(supp_x, [.5, 0, 0]))), model_x)
# mean_y_x = 1 / (2 + np.sin(2*np.pi * supp_x))
# model = rand_models.DataConditional(list(map(mean_to_rv, poly(supp_x, [.5, 0, 0]))), model_x)


# model = rand_models.NormalLinear(weights=np.ones(2), basis_y_x=None, cov_y_x=.1,
#                                  model_x=rand_elements.Normal(0, 10), rng=None)
# model = rand_models.ClassConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                  ['a', 'b'], p_y=None)


# model = bayes_models.Dirichlet(model, alpha_0=10)


#
do_bayes = True
# do_bayes = False
if do_bayes:
    model_eval = bayes_models.Dirichlet(model, alpha_0=10)
    opt_predictor = BayesRegressor(model_eval, name=r'$f^*$')
else:
    model_eval = model
    opt_predictor = ModelRegressor(model_eval, name=r'$f_{\Theta}$')

# if isinstance(model, rand_models.Base):
#     opt_predictor = ModelRegressor(model, name=r'$f_{\Theta}$')
# elif isinstance(model, bayes_models.Base):
#     opt_predictor = BayesRegressor(model, name=r'$f^*$')
# else:
#     raise TypeError



# Bayesian learners

w_prior = [.5, 0]
# w_prior = [.5, 0, .5]


# prior_mean = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                      supp_x=[0, .5], p_x=None)

# prior_mean = rand_models.DataConditional(list(map(mean_to_rv, poly(supp_x, w_prior))), model_x)

prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=1000, model_x=rand_elements.Beta())

dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=10), name='Dir')

# dir_params = None
# dir_params = {'alpha_0': [1, 10, 100]}
dir_params = {'alpha_0': [.1, 50]}
# dir_params = {'alpha_0': .001 + np.arange(0, 80, 5)}


# Normal learner
norm_predictor = BayesRegressor(bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=10 * np.eye(len(w_prior)),
                                                          basis_y_x=None, cov_y_x=.1,
                                                          model_x=model_x), name='Norm')

# norm_params = None
norm_params = {'prior_cov': [10, 0.05]}
# norm_params = {'prior_cov': [10]}


# Plotting

# n_train = 10
# n_train = [0, 10, 20]
n_train = np.arange(0, 100, 10)

# print(dir_predictor.risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=20000, verbose=True, rng=None))
# dir_predictor.plot_risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=5000, verbose=True, rng=None)


temp = [
    (opt_predictor, None),
    (norm_predictor, norm_params),
    (dir_predictor, dir_params),
]

predictors, params = list(zip(*temp))


plot_risk_eval_sim_compare(predictors, model_eval, params, n_train=n_train, n_test=1, n_mc=500,
                           verbose=True, ax=None, rng=None)
# plot_risk_eval_comp_compare(predictors, model_eval, params, n_train, n_test=1, verbose=False, ax=None)

# plot_predict_stats_compare(predictors, model_eval, params, x=None, n_train=n_train, n_mc=300, do_std=True,
#                            verbose=True, ax=None, rng=None)


# print(f"\nAnalytical Risk = {opt_predictor.evaluate_comp(n_train=n_train)}")

# if isinstance(model, rand_models.Base):
#     risk_an = opt_predictor.risk_min()
#     print(f"Min risk = {risk_an}")
# elif isinstance(model, bayes_models.Base):
#     risk_an = opt_predictor.bayes_risk_min(n_train)
#     print(f"Min Bayes risk = {risk_an}")
# else:
#     raise TypeError

print('Done')
