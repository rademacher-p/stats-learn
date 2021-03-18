"""
Main.
"""

from pathlib import Path
import pickle
from time import strftime
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from thesis.bayes import models as bayes_models
from thesis.predictors import (ModelRegressor, BayesRegressor, plot_risk_eval_sim_compare, plot_predict_stats_compare,
                               risk_eval_sim_compare, plot_risk_eval_comp_compare, plot_risk_disc)
from thesis.random import elements as rand_elements, models as rand_models
from thesis.preprocessing import discretizer
from thesis.util.base import all_equal


# plt.style.use('seaborn')
# plt.style.use(['science'])


#%% Sim
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

alpha_y_x_beta = 1/var_y_x_const - 1
# alpha_y_x_d = (1-var_y_x_const) / (np.float64(var_y_x_const) - 1/(n_x-1))
alpha_y_x_d = alpha_y_x_beta / (1 - 1/(n_x-1)/np.float64(var_y_x_const))


# True model

# model = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)],
#                                                 supp_x=[0, .5], p_x=None)

# w_model = [0, 0, 1]
# w_model = [0, 0, 0, 0, 1]
# w_model = [.3, 0., .4]
w_model = [.5, 0, 0]


def nonlinear_model(x):
    return 1 / (2 + np.sin(2*np.pi * x))
    # return 1 / (1e-9 + 2 + np.sin(2*np.pi * x))


# model = rand_models.DataConditional.from_finite(poly_mean_to_models(n_x, alpha_y_x_d, w_model),
#                                                 supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)
# model = rand_models.DataConditional.from_finite(func_mean_to_models(n_x, alpha_y_x_d, nonlinear_model),
#                                                 supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)

# model = rand_models.BetaLinear(weights=w_model, basis_y_x=None, alpha_y_x=alpha_y_x_beta)
model = rand_models.BetaLinear(weights=[1], basis_y_x=[nonlinear_model], alpha_y_x=alpha_y_x_beta)

# model = rand_models.NormalLinear(weights=np.ones(2), basis_y_x=None, cov_y_x=.1, model_x=rand_elements.Normal(0, 10))


do_bayes = False
# do_bayes = True
if do_bayes:
    model_eval = bayes_models.Dirichlet(model, alpha_0=4e2)
    opt_predictor = BayesRegressor(model_eval, name=r'$f^*$')
else:
    model_eval = model
    opt_predictor = ModelRegressor(model_eval, name=r'$f_{\Theta}(\theta)$')


# Bayesian learners

w_prior = [.5, 0]
# w_prior = [.5, 0, 0]


# Dirichlet learner
proc_funcs = []

# prior_mean = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.9, .9)],
#                                                      supp_x=[0, .5], p_x=None)

# prior_mean = rand_models.DataConditional.from_finite(poly_mean_to_models(n_x, alpha_y_x_d, w_prior),
#                                                      supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)


# prior_mean_x = rand_elements.Beta()

n_t = 4
_temp = np.full(n_t, 2)
_temp[[0, -1]] = 1  # first/last half weight due to rounding discretizer and uniform marginal model
prior_mean_x = rand_elements.Finite(np.linspace(0, 1, n_t, endpoint=True), p=_temp / _temp.sum())
proc_funcs.append(discretizer(prior_mean_x.supp))

prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x_beta, model_x=prior_mean_x)

_name = r'$\mathrm{Dir}$'
if len(proc_funcs) > 0:
    _name += r', $|\mathcal{T}| = __card__$'.replace('__card__', str(prior_mean_x.supp.size))

dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=10),
                               space=model.space, proc_funcs=proc_funcs,
                               name=_name,
                               )

# dir_params = None
dir_params = {'alpha_0': [10, 1000]}
# dir_params = {'alpha_0': [10]}
# dir_params = {'alpha_0': [.01, 100]}
# dir_params = {'alpha_0': [40, 400, 4000]}
# dir_params = {'alpha_0': 1e-6 + np.linspace(0, 20, 100)}
# dir_params = {'alpha_0': np.logspace(-1., 5., 60)}
# dir_params = {'alpha_0': np.logspace(-3., 3., 100)}

if do_bayes:  # add true bayes model concentration
    if model_eval.alpha_0 not in dir_params['alpha_0']:
        dir_params['alpha_0'] = np.sort(np.concatenate((dir_params['alpha_0'], [model_eval.alpha_0])))


###
# n_t_iter = [4, 128, 4096]
# n_t_iter = [4, 16, 32, 64, 128]
# n_t_iter = [2, 4, 8, 16]
# n_t_iter = 2 ** np.arange(1, 6)
n_t_iter = list(range(1, 33, 1))
# n_t_iter = list(range(4, 64, 4))


scale_alpha = False
# scale_alpha = True

dir_predictors = []
dir_params_full = [deepcopy(dir_params) for __ in n_t_iter]
for n_t, _params in zip(n_t_iter, dir_params_full):
    _temp = np.full(n_t, 2)
    _temp[[0, -1]] = 1  # first/last half weight due to rounding discretizer and uniform marginal model
    prior_mean_x = rand_elements.Finite(np.linspace(0, 1, n_t, endpoint=True), p=_temp / _temp.sum())
    prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x_beta, model_x=prior_mean_x)

    dir_predictors.append(BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=0.01),
                                         space=model.space, proc_funcs=[discretizer(prior_mean_x.supp)],
                                         name=r'$\mathrm{Dir}$, $|\mathcal{T}| = card$'.replace('card', str(n_t)),
                                         ))

    if scale_alpha and _params is not None:
        _params['alpha_0'] *= n_t


# Normal learner
norm_predictor = BayesRegressor(bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=100 * np.eye(len(w_prior)),
                                                          basis_y_x=None, cov_y_x=.1,
                                                          model_x=model.model_x), name=r'$\mathcal{N}$')

# norm_params = None
norm_params = {'prior_cov': [.1, .001]}
# norm_params = {'prior_cov': [.1]}
# norm_params = {'prior_cov': [100, .01]}

# Plotting

# n_train = 400
# n_train = [0, 4, 40, 400]
# n_train = [0, 800, 4000]
# n_train = [0, 100, 200, 400, 800]
# n_train = np.arange(0, 650, 50)
n_train = np.arange(0, 4500, 500)
# n_train = np.concatenate((np.arange(0, 250, 50), np.arange(200, 4500, 500)))


# print(dir_predictor.risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=20000, verbose=True, rng=None))
# dir_predictor.plot_risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=5000, verbose=True, rng=None)


temp = [
    # (opt_predictor, None),
    (dir_predictor, dir_params),
    # *(zip(dir_predictors, dir_params_full)),
    # (norm_predictor, norm_params),
]

# TODO: discrete plot for predict stats
# TODO: make latex use optional

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{upgreek} \usepackage{bm}")

predictors, params = list(zip(*temp))

# TODO: efficient sequential ops for loss, mean, etc.?

plot_risk_eval_sim_compare(predictors, model_eval, params, n_train, n_mc=500, verbose=True, ax=None, rng=None)
# plot_risk_eval_comp_compare(predictors, model_eval, params, n_train, verbose=False, ax=None)

# plot_predict_stats_compare(predictors, model_eval, params, x=None, n_train=n_train, n_mc=50000,
#                            do_std=True, verbose=True, ax=None, rng=None)


# plot_risk_disc(predictors, model_eval, params, n_train, n_test=1, n_mc=500, verbose=True, ax=None, rng=None)


# Find localization minimum
do_argmin = False
# do_argmin = True
ax = plt.gca()
if ax.get_xlabel() == r'$\alpha_0$':
    ax.set_xscale('log')
    lines = ax.get_lines()
    for line in lines:
        x_, y_ = line.get_data()
        idx = y_.argmin()
        x_i, y_i = x_[idx], y_[idx]
        if scale_alpha:
            label = line.get_label()
            _n_t = int(label[label.find('=')+1:-1])
            line.set_data(x_ / _n_t, y_)
            x_i /= _n_t
        if do_argmin:
            ax.plot(x_i, y_i, marker='.', markersize=8, color=line.get_color())
    if scale_alpha:
        ax.set_xlabel(r'$\alpha_0 / |\mathcal{T}|$ ')
        _vals = dir_params['alpha_0']
        ax.set_xlim((_vals.min(), _vals.max()))


#%% Save image and Figure
time_str = strftime('%Y-%m-%d_%H-%M-%S')
image_path = Path('./images/temp/')

fig = plt.gcf()
fig.savefig(image_path.joinpath(f"{time_str}.png"))
with open(image_path.joinpath(f"{time_str}.mpl"), 'wb') as fid:
    pickle.dump(fig, fid)

print('Done')


#%%
# print(f"\nAnalytical Risk = {opt_predictor.evaluate_comp(n_train=n_train)}")

# if isinstance(model, rand_models.Base):
#     risk_an = opt_predictor.risk_min()
#     print(f"Min risk = {risk_an}")
# elif isinstance(model, bayes_models.Base):
#     risk_an = opt_predictor.bayes_risk_min(n_train)
#     print(f"Min Bayes risk = {risk_an}")
# else:
#     raise TypeError
