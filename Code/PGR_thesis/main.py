"""
Main.
"""

from pathlib import Path
import pickle
from time import strftime
from copy import deepcopy
from math import prod

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from thesis.bayes import models as bayes_models
from thesis.predictors import (ModelRegressor, BayesRegressor, plot_risk_eval_sim_compare, plot_predict_stats_compare,
                               risk_eval_sim_compare, plot_risk_eval_comp_compare, plot_risk_disc, SKLWrapper)
from thesis.random import elements as rand_elements, models as rand_models
from thesis.preprocessing import discretizer, prob_disc
from thesis.util.plotting import box_grid


np.set_printoptions(precision=3)

# plt.style.use('seaborn')
# plt.style.use(['science'])

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{upgreek} \usepackage{bm}")

seed = None
# seed = 12345


#%% Model
def poly_mean_to_models(n, alpha_0, weights):
    return func_mean_to_models(n, alpha_0, lambda x: sum(w * x ** i for i, w in enumerate(weights)))


def func_mean_to_models(n, alpha_0, func):
    x_supp = np.linspace(0, 1, n, endpoint=True)
    if np.isinf(alpha_0):
        return [rand_elements.EmpiricalScalar(func(_x), n-1) for _x in x_supp]
    else:
        return [rand_elements.DirichletEmpiricalScalar(func(_x), alpha_0, n-1) for _x in x_supp]


n_x = 128

# var_y_x_const = 1 / (n_x-1)
var_y_x_const = 1/5

alpha_y_x_d = (1-var_y_x_const) / (np.float64(var_y_x_const) - 1/(n_x-1))
alpha_y_x_beta = 1/var_y_x_const - 1


# True model

# model_x = rand_elements.Finite([0, .5], p=None)
# model = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)], model_x)


shape_x = ()
# shape_x = (2,)

# w_model = [.5]
w_model = [0, 1]


def nonlinear_model(x):
    # return 1 / (2 + np.sin(2*np.pi * x))
    axis = tuple(range(-len(shape_x), 0))
    return 1 / (2 + np.sin(2 * np.pi * x.mean(axis)))


# supp_x = box_grid(np.broadcast_to([0, 1], (*shape_x, 2)), n_x, endpoint=True)
# _temp = np.ones(prod(shape_x)*(n_x,))
# model_x = rand_elements.Finite(supp_x, p=_temp/_temp.sum())
# # model = rand_models.DataConditional(poly_mean_to_models(n_x, alpha_y_x_d, w_model), model_x)
# model = rand_models.DataConditional(func_mean_to_models(n_x, alpha_y_x_d, nonlinear_model), model_x)

model_x = rand_elements.Uniform(np.broadcast_to([0, 1], (*shape_x, 2)))
# model = rand_models.BetaLinear(weights=w_model, basis_y_x=None, alpha_y_x=alpha_y_x_beta, model_x=model_x)
model = rand_models.BetaLinear(weights=[1], basis_y_x=[nonlinear_model], alpha_y_x=alpha_y_x_beta, model_x=model_x)

# model = rand_models.NormalLinear(weights=w_model, basis_y_x=None, cov_y_x=.1, model_x=model_x)


do_bayes = False
# do_bayes = True
if do_bayes:
    model_eval = bayes_models.Dirichlet(deepcopy(model), alpha_0=4e2)
    opt_predictor = BayesRegressor(model_eval, name=r'$f^*$')
else:
    model_eval = model
    opt_predictor = ModelRegressor(model_eval, name=r'$f_{\Theta}(\theta)$')

model_eval.rng = seed


#%% Bayesian learners

# w_prior = [.5, 0]
w_prior = [0, 1]
# w_prior = [.5, 0, 0, 0, 0, 0, 0]


# Dirichlet learner
proc_funcs = []

# prior_mean = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.9, .9)], model_x)
# prior_mean = rand_models.DataConditional(poly_mean_to_models(n_x, alpha_y_x_d, w_prior), model_x)


# prior_mean_x = deepcopy(model_x)

n_t = 16
supp_x = box_grid(model_x.lims, n_t, endpoint=True)
# _temp = np.ones(model_x.size*(n_t,))
_temp = prob_disc(model_x.size*(n_t,))
prior_mean_x = rand_elements.Finite(supp_x, p=_temp/_temp.sum())
proc_funcs.append(discretizer(supp_x.reshape(-1, *model_x.shape)))

prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x_beta, model_x=prior_mean_x)


_name = r'$\mathrm{Dir}$'
if len(proc_funcs) > 0:
    _card = str(n_t)
    if model_x.size > 1:
        _card += f"^{model_x.size}"
    _name += r', $|\mathcal{T}| = __card__$'.replace('__card__', _card)

dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=10), proc_funcs=proc_funcs, name=_name)

dir_params = {}
# dir_params = {'alpha_0': [10, 1000]}
# dir_params = {'alpha_0': [10]}
# dir_params = {'alpha_0': [.01, 100]}
# dir_params = {'alpha_0': [40, 400, 4000]}
# dir_params = {'alpha_0': 1e-6 + np.linspace(0, 20, 100)}
# dir_params = {'alpha_0': np.logspace(-0., 5., 60)}
# dir_params = {'alpha_0': np.logspace(-3., 3., 100)}

if do_bayes:  # add true bayes model concentration
    if model_eval.alpha_0 not in dir_params['alpha_0']:
        dir_params['alpha_0'] = np.sort(np.concatenate((dir_params['alpha_0'], [model_eval.alpha_0])))


# Normal learner
norm_predictor = BayesRegressor(bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=.1,
                                                          basis_y_x=None, cov_y_x=.1,
                                                          model_x=model_x), name=r'$\mathcal{N}$')

norm_params = {}
# norm_params = {'prior_cov': [.1, .001]}
# norm_params = {'prior_cov': [1000000]}
# norm_params = {'prior_cov': [100, .001]}
# norm_params = {'prior_cov': np.logspace(-7., 3., 60)}


#%% Scikit-Learn
# skl_estimator, _name = LinearRegression(), 'LR'
# skl_estimator, _name = SGDRegressor(max_iter=1000, tol=None), 'SGD'
skl_estimator, _name = MLPRegressor(hidden_layer_sizes=[100 for _ in range(4)], solver='adam', alpha=1e-4,
                                    max_iter=2000, tol=1e-8, verbose=False), 'MLP'


# TODO: try Adaboost, RandomForest, GP, BayesianRidge, KNeighbors, SVR

# skl_estimator = Pipeline([('scaler', StandardScaler()), ('regressor', skl_estimator)])
skl_predictor = SKLWrapper(skl_estimator, space=model.space, name=_name)


#%% Results

# n_train = 40
# n_train = [1, 4, 40, 400]
# n_train = [0, 200, 400, 600]
# n_train = [0, 400, 4000]
# n_train = [100, 200]
# n_train = np.arange(0, 320, 20)
n_train = np.arange(0, 55, 5)
# n_train = np.arange(0, 4500, 500)
# n_train = np.concatenate((np.arange(0, 250, 50), np.arange(200, 4050, 50)))


temp = [
    (opt_predictor, None),
    (dir_predictor, dir_params),
    # *(zip(dir_predictors, dir_params_full)),
    # (norm_predictor, norm_params),
    (skl_predictor, None),
]
predictors, params = zip(*temp)


# TODO: add logic based on which parameters can be changed while preserving learner state!!
# TODO: train/test loss results?

plot_risk_eval_sim_compare(predictors, model_eval, params, n_train, n_test=100, n_mc=50, verbose=True)
# plot_predict_stats_compare(predictors, model_eval, params, n_train, n_mc=100, x=None, do_std=True, verbose=True)

# d = model.rvs(10)
# ax = model.space['x'].make_axes()
# norm_predictor.plot_fit(d, ax=ax)
# skl_predictor.plot_fit(d, ax=ax)
# ax.set_ylim((0, 1))


# Save image and Figure
time_str = strftime('%Y-%m-%d_%H-%M-%S')
image_path = Path('./images/temp/')

fig = plt.gcf()
fig.savefig(image_path.joinpath(f"{time_str}.png"))
with open(image_path.joinpath(f"{time_str}.mpl"), 'wb') as fid:
    pickle.dump(fig, fid)

print('Done')


#%% Deprecated

# ###
# n_t_iter = [4, 128, 4096]
# # n_t_iter = [4, 16, 32, 64, 128]
# # n_t_iter = [2, 4, 8, 16]
# # n_t_iter = 2 ** np.arange(1, 14)
# # n_t_iter = list(range(1, 33, 1))
# # n_t_iter = list(range(4, 64, 4))
#
#
# scale_alpha = False
# # scale_alpha = True
#
# dir_predictors = []
# dir_params_full = [deepcopy(dir_params) for __ in n_t_iter]
# for n_t, _params in zip(n_t_iter, dir_params_full):
#     _temp = np.full(n_t, 2)
#     _temp[[0, -1]] = 1  # first/last half weight due to rounding discretizer and uniform marginal model
#     prior_mean_x = rand_elements.Finite(np.linspace(0, 1, n_t, endpoint=True), p=_temp / _temp.sum())
#     prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x_beta, model_x=prior_mean_x)
#
#     dir_predictors.append(BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=0.01),
#                                          space=model.space, proc_funcs=[discretizer(prior_mean_x.supp)],
#                                          name=r'$\mathrm{Dir}$, $|\mathcal{T}| = card$'.replace('card', str(n_t)),
#                                          ))
#
#     if scale_alpha and _params is not None:
#         _params['alpha_0'] *= n_t

# plot_risk_disc(predictors, model_eval, params, n_train, n_test=1, n_mc=50000, verbose=True, ax=None)
# plt.xscale('log', base=2)

# # Scale alpha axis, find localization minimum
# do_argmin = False
# # do_argmin = True
# ax = plt.gca()
# if ax.get_xlabel() == r'$\alpha_0$':
#     ax.set_xscale('log')
#     lines = ax.get_lines()
#     for line in lines:
#         x_, y_ = line.get_data()
#         idx = y_.argmin()
#         x_i, y_i = x_[idx], y_[idx]
#         if scale_alpha:
#             label = line.get_label()
#             _n_t = int(label[label.find('=')+1:-1])
#             line.set_data(x_ / _n_t, y_)
#             x_i /= _n_t
#         if do_argmin:
#             ax.plot(x_i, y_i, marker='.', markersize=8, color=line.get_color())
#     if scale_alpha:
#         ax.set_xlabel(r'$\alpha_0 / |\mathcal{T}|$ ')
#         _vals = dir_params['alpha_0']
#         ax.set_xlim((_vals.min(), _vals.max()))


# print(dir_predictor.risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=20000, verbose=True))
# dir_predictor.plot_risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=5000, verbose=True)

# plot_risk_eval_comp_compare(predictors, model_eval, params, n_train, verbose=False, ax=None)

# print(f"\nAnalytical Risk = {opt_predictor.evaluate_comp(n_train=n_train)}")

# if isinstance(model, rand_models.Base):
#     risk_an = opt_predictor.risk_min()
#     print(f"Min risk = {risk_an}")
# elif isinstance(model, bayes_models.Base):
#     risk_an = opt_predictor.bayes_risk_min(n_train)
#     print(f"Min Bayes risk = {risk_an}")
# else:
#     raise TypeError
