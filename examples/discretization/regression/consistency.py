# TODO: CLEANUP


from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from stats_learn.util.base import get_now
from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors.base import ModelRegressor, BayesRegressor
from stats_learn.util import funcs
from stats_learn import results
from stats_learn.util.math import prob_disc
from stats_learn.util.data_processing import make_discretizer


plt.style.use('../../../images/style.mplstyle')

# seed = None
seed = 12345

# log_path = None
# img_path = None

# TODO: remove path stuff and image names below before release
# base_path = 'consistency_temp/'
base_path = __file__[__file__.rfind('/')+1:].removesuffix('.py') + '_temp/'
log_path = base_path + 'log.md'
img_dir = base_path + f'{get_now()}/'


#%% Model and optimal predictor
nonlinear_model = funcs.make_inv_trig()
var_y_x_const = 1/5

alpha_y_x_beta = 1/var_y_x_const - 1
model_x = rand_elements.Uniform([0, 1])
model = rand_models.BetaLinear(weights=[1], basis_y_x=[nonlinear_model], alpha_y_x=alpha_y_x_beta, model_x=model_x)

opt_predictor = ModelRegressor(model, name=r'$f_{\Theta}(\theta)$')


#%% Learners
w_prior = [.5, 0]

# Dirichlet

# dir_params = {'alpha_0': [10]}
dir_params = {'alpha_0': np.logspace(-3, 3, 60)}

# n_t_iter = [4, 128, 4096]
n_t_iter = [2, 4, 8, 16]

# scale_alpha = False
scale_alpha = True  # interpret `alpha_0` parameter as normalized w.r.t. discretization cardinality

dir_predictors = []
dir_params_full = [deepcopy(dir_params) for __ in n_t_iter]
for n_t, _params in zip(n_t_iter, dir_params_full):
    supp_t = np.linspace(*model_x.lims, n_t, endpoint=True)
    _temp = prob_disc(supp_t.shape)

    prior_mean_x = rand_elements.DataEmpirical(supp_t, counts=_temp, space=model_x.space)
    prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x_beta,
                                        model_x=prior_mean_x)
    dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)

    dir_predictor = BayesRegressor(dir_model, space=model.space, proc_funcs=[make_discretizer(supp_t)],
                                   name=r'$\mathrm{Dir}$, $|\mathcal{T}| = card$'.replace('card', str(n_t)))

    dir_predictors.append(dir_predictor)

    if scale_alpha and _params is not None:
        _params['alpha_0'] *= n_t


# Normal-prior LR
norm_model = bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=.1, cov_y_x=.1, model_x=model_x,
                                       allow_singular=True)
norm_predictor = BayesRegressor(norm_model, space=model.space, name=r'$\mathcal{N}$')

norm_params = {'prior_cov': [.1, .001]}


#
temp = [
    (opt_predictor, None),
    *zip(dir_predictors, dir_params_full),
    (norm_predictor, norm_params),
]
predictors, params = zip(*temp)


#%% Results
n_test = 1000
n_mc = 50

# Sample regressor realizations
n_train = 30
d = model.rvs(n_train + n_test, rng=seed)
d_train, d_test = np.split(d, [n_train])
x = np.linspace(0, 1, 10000)

img_path = img_dir + 'fit.png'
loss_full = results.plot_fit_compare(predictors, d_train, d_test, params, x, log_path=log_path, img_path=img_path)

# Prediction mean/variance, comparative
n_train = 400

img_path = img_dir + 'predict_T.png'
y_stats_full, loss_full = results.assess_compare(predictors, model, params, n_train, n_test, n_mc,
                                                 stats=('mean', 'std'), verbose=True, plot_stats=True, print_loss=True,
                                                 log_path=log_path, img_path=img_path, rng=seed)

# Dirichlet-based prediction mean/variance, varying N
n_train = [0, 400, 4000]
_t = 4
idx = n_t_iter.index(_t)

img_path = img_dir + f'predict_N_T{_t}.png'
y_stats_full, loss_full = dir_predictors[idx].assess(model, {'alpha_0': [1000]}, n_train, n_test, n_mc,
                                                     stats=('mean', 'std'),
                                                     verbose=True, plot_stats=True, print_loss=True,
                                                     log_path=log_path, img_path=img_path, rng=seed)

# Squared-Error vs. training data volume N
n_train = np.arange(0, 4500, 500)

img_path = img_dir + 'risk_N_leg_T.png'
y_stats_full, loss_full = results.assess_compare(predictors, model, params, n_train, n_test, n_mc, verbose=True,
                                                 plot_loss=True, print_loss=True, log_path=log_path,
                                                 img_path=img_path, rng=seed)

# Squared-Error vs. prior localization alpha_0
n_train = 4

img_path = img_dir + 'risk_a0norm_leg_T.png'
y_stats_full, loss_full = results.assess_compare(dir_predictors, model, dir_params_full, n_train, n_test, n_mc,
                                                 verbose=True, plot_loss=True, print_loss=True, log_path=log_path,
                                                 img_path=img_path, rng=seed)


do_argmin = False
# do_argmin = True
ax = plt.gca()
if ax.get_xlabel() == r'$\alpha_0$':  # scale alpha axis, find localization minimum
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
        ax.set_xlim((min(_vals), max(_vals)))
