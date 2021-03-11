"""
Main.
"""

from pathlib import Path
import pickle
from time import strftime

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
# plt.rcParams['text.usetex'] = True


# %% Sim

def poly_mean_to_models(n, weights):
    return func_mean_to_models(n, lambda x: sum(w * x ** i for i, w in enumerate(weights)))


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

# model = rand_models.DataConditional.from_finite(poly_mean_to_models(n_x, w_model),
#                                                 supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)
# model = rand_models.DataConditional.from_finite(func_mean_to_models(n_x, lambda x: 1 / (2 + np.sin(2*np.pi * x))),
#                                                 supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)

# model = rand_models.BetaLinear(weights=w_model, basis_y_x=None, alpha_y_x=126, model_x=rand_elements.Beta())
model = rand_models.BetaLinear(weights=[1], basis_y_x=[lambda x: 1 / (2 + np.sin(2 * np.pi * x))], alpha_y_x=6,
                               model_x=rand_elements.Beta())

# model = rand_models.NormalLinear(weights=np.ones(2), basis_y_x=None, cov_y_x=.1, model_x=rand_elements.Normal(0, 10))


# do_bayes = True
do_bayes = False
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

# prior_mean = rand_models.DataConditional.from_finite([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.9, .9)],
#                                                      supp_x=[0, .5], p_x=None)

# prior_mean = rand_models.DataConditional.from_finite(poly_mean_to_models(n_x, w_prior),
#                                                      supp_x=np.linspace(0, 1, n_x, endpoint=True), p_x=None)


# prior_mean_x = rand_elements.Beta()

_temp = np.full(n_x, 2)
_temp[[0, -1]] = 1  # first/last half weight due to rounding discretizer and uniform marginal model
prior_mean_x = rand_elements.Finite(np.linspace(0, 1, n_x, endpoint=True), p=_temp / _temp.sum())
proc_funcs.append(discretizer(prior_mean_x.supp))

prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=126, model_x=prior_mean_x)

_name = r'$\mathrm{Dir}$'
if len(proc_funcs) > 0:
    _name += r', $|\mathcal{T}| = __card__$'.replace('__card__', str(n_x))
dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=100),
                               space=model.space, proc_funcs=proc_funcs,
                               name=_name,
                               # name='$\mathrm{Dir}$, $|\mathcal{T}| = card$'.replace('card', str(n_x)),
                               )


##
dir_predictors = []
# n_x_iter = [4, 128, 4096]
# n_x_iter = [64, 128, 256]
# n_x_iter = 2 ** np.arange(1, 6)
n_x_iter = list(range(2, 33, 2))
for n_x in n_x_iter:
    _temp = np.full(n_x, 2)
    _temp[[0, -1]] = 1  # first/last half weight due to rounding discretizer and uniform marginal model
    prior_mean_x = rand_elements.Finite(np.linspace(0, 1, n_x, endpoint=True), p=_temp / _temp.sum())
    prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=126, model_x=prior_mean_x)

    dir_predictors.append(BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=100),
                                         space=model.space, proc_funcs=[discretizer(prior_mean_x.supp)],
                                         name='$\mathrm{Dir}$, $|\mathcal{T}| = card$'.replace('card', str(n_x)),
                                         ))


# dir_params = None
# dir_params = {'alpha_0': [1, 100, 10000]}
dir_params = {'alpha_0': [.01, 100]}
# dir_params = {'alpha_0': [0.01]}
# dir_params = {'alpha_0': 1e-6 + np.linspace(0, 20, 100)}
# dir_params = {'alpha_0': np.logspace(-0., 6., 40)}


# Normal learner
norm_predictor = BayesRegressor(bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=100 * np.eye(len(w_prior)),
                                                          basis_y_x=None, cov_y_x=.1,
                                                          model_x=model.model_x), name=r'$\mathcal{N}$')

# norm_params = None
# norm_params = {'prior_cov': [10, 0.05]}
# norm_params = {'prior_cov': [100, .01]}
norm_params = {'prior_cov': [100]}

# Plotting

n_train = 100
# n_train = [0, 10, 50, 100]
# n_train = [0, 100, 200]
# n_train = [0, 2, 8]
# n_train = np.arange(0, 650, 50)
# n_train = np.arange(0, 5500, 500)


# print(dir_predictor.risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=20000, verbose=True, rng=None))
# dir_predictor.plot_risk_eval_sim(model, dir_params, n_train, n_test=1, n_mc=5000, verbose=True, rng=None)


temp = [
    # (opt_predictor, None),
    # (dir_predictor, dir_params),
    *((pr, dir_params) for pr in dir_predictors),
    # (norm_predictor, norm_params),
]

# TODO: discrete plot for predict stats
# TODO: make latex use optional

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{upgreek} \usepackage{bm}")

predictors, params = list(zip(*temp))

# plot_risk_eval_sim_compare(predictors, model_eval, params, n_train, n_mc=500, verbose=True, ax=None, rng=None)
# plot_risk_eval_comp_compare(predictors, model_eval, params, n_train, verbose=False, ax=None)

# plot_predict_stats_compare(predictors, model_eval, params, x=None, n_train=n_train, n_mc=300,
#                            do_std=True, verbose=True, ax=None, rng=None)


plot_risk_disc(predictors, model_eval, params, n_train, n_test=1, n_mc=500, verbose=True, ax=None, rng=None)




# Find localization minimum
ax = plt.gca()
if ax.get_xlabel() == r'$\alpha_0$':
    ax.set_xscale('log')
    lines = ax.get_lines()
    for line in lines:
        x_, y_ = line.get_data()
        idx = y_.argmin()
        ax.plot(x_[idx], y_[idx], 'k*', markersize=8)


#%% Save image and Figure
time_str = strftime('%Y-%m-%d_%H-%M-%S')
image_path = Path('./images/temp/')

fig = plt.gcf()
fig.savefig(image_path.joinpath(f"{time_str}.png"))
with open(image_path.joinpath(f"{time_str}.mpl"), 'wb') as fid:
    pickle.dump(fig, fid)


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
