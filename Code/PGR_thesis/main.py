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
from thesis.preprocessing import discretizer
from thesis.util import spaces

# plt.style.use('seaborn')
# plt.style.use(['science'])
# plt.rcParams['text.usetex'] = True


#%% Sim

def poly_mean_to_models(n, weights):
    return func_mean_to_models(n, lambda x_: sum(w * x_ ** i for i, w in enumerate(weights)))


def func_mean_to_models(n, func):
    return [rand_elements.EmpiricalScalar(func(x_i), n - 1) for x_i in np.linspace(0, 1, n, endpoint=True)]


n_x = 128


# True model

# model = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                 supp_x=[0, .5], p_x=None)


# w_model = [0, 0, 1]
# w_model = [0, 0, 0, 0, 1]
# w_model = [.3, 0., .4]
w_model = [.5, 0, 0]


model = rand_models.DataConditional.from_finite(poly_mean_to_models(n_x, w_model),
                                                supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)
# model = rand_models.DataConditional.from_finite(func_mean_to_models(n_x, lambda x: 1 / (2 + np.sin(2*np.pi * x))),
#                                                 supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)

# model = rand_models.BetaLinear(weights=w_model, basis_y_x=None, alpha_y_x=100, model_x=rand_elements.Beta())
# model = rand_models.BetaLinear(weights=[1], basis_y_x=[lambda x: 1 / (2 + np.sin(2*np.pi * x))], alpha_y_x=100,
#                                model_x=rand_elements.Beta())


# model = rand_models.NormalLinear(weights=np.ones(2), basis_y_x=None, cov_y_x=.1, model_x=rand_elements.Normal(0, 10))


do_bayes = True
# do_bayes = False
if do_bayes:
    model_eval = bayes_models.Dirichlet(model, alpha_0=100)
    opt_predictor = BayesRegressor(model_eval, name=r'$f^*$')
else:
    model_eval = model
    opt_predictor = ModelRegressor(model_eval, name=r'$f_{\Theta}(\theta)$')


# Bayesian learners

w_prior = [.5, 0]
# w_prior = [.5, 0, 0]


# Dirichlet learner
proc_funcs = []

# prior_mean = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.9, .1)],
#                                                      supp_x=[0, .5], p_x=None)

prior_mean = rand_models.DataConditional.from_finite(poly_mean_to_models(n_x, w_prior),
                                                     supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)
# prior_mean = rand_models.DataConditional.from_finite(func_mean_to_models(n_x, lambda x: 1 / (2 + np.sin(2*np.pi * x))),
#                                                      supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)

# prior_mean_x = rand_elements.Beta()

# prior_mean_x = rand_elements.Mixture([rand_elements.DataEmpirical(np.linspace(0, 1, n_x, endpoint=True),
#                                                                   counts=np.ones(n_x), space=model.space['x']),
#                                       rand_elements.Beta()],
#                                      weights=[1000, 1])
# prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=100, model_x=prior_mean_x)
# proc_funcs.append(discretizer(prior_mean_x.dists[0].data['x']))


dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=10), proc_funcs=proc_funcs,
                               name='$\mathrm{Dir}$')

# dir_params = None
# dir_params = {'alpha_0': [1, 100, 10000]}
# dir_params = {'alpha_0': [.1, 50]}
# dir_params = {'alpha_0': [.01, 100]}
# dir_params = {'alpha_0': [100]}
# dir_params = {'alpha_0': 1e-6 + np.linspace(0, 100, 100)}
# dir_params = {'alpha_0': 1e-6 + np.concatenate((np.linspace(0, 20, 100), np.linspace(20, 100000, 1000)))}
dir_params = {'alpha_0': np.logspace(-1., 5., 100)}

# Normal learner
norm_predictor = BayesRegressor(bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=100 * np.eye(len(w_prior)),
                                                          basis_y_x=None, cov_y_x=.1,
                                                          model_x=prior_mean.model_x), name='$\mathcal{N}$')

# norm_params = None
# norm_params = {'prior_cov': [10, 0.05]}
norm_params = {'prior_cov': [100, .01]}
# norm_params = {'prior_cov': [100]}


# Plotting

# n_train = 100
n_train = [0, 100, 1000]
# n_train = [0, 5, 10]
# n_train = np.arange(0, 110, 10)
# n_train = np.arange(0, 1100, 100)


# print(dir_predictor.risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=20000, verbose=True, rng=None))
# dir_predictor.plot_risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=5000, verbose=True, rng=None)


temp = [
    # (opt_predictor, None),
    (dir_predictor, dir_params),
    # (norm_predictor, norm_params),
]

# TODO: discrete plot for predict stats
# TODO: save fig, png and object!
# TODO: redo SSP p_dir fig

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{upgreek}")

predictors, params = list(zip(*temp))

plot_risk_eval_sim_compare(predictors, model_eval, params, n_train=n_train, n_test=1, n_mc=50000,
                           verbose=True, ax=None, rng=None)
# plot_risk_eval_comp_compare(predictors, model_eval, params, n_train, n_test=1, verbose=False, ax=None)

# plot_predict_stats_compare(predictors, model_eval, params, x=None, n_train=n_train, n_mc=50000,
#                            do_std=True, verbose=True, ax=None, rng=None)


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
