"""
Main.
"""
import math
from pathlib import Path
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor

import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything

from stats_learn.bayes import models as bayes_models
from stats_learn.predictors import ModelRegressor, BayesRegressor, SKLWrapper
from stats_learn.util import funcs
from stats_learn.util import results
from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.util.plotting import box_grid
from stats_learn.util.torch import LitMLP, LitWrapper

np.set_printoptions(precision=3)

# plt.style.use('seaborn')
# plt.style.use(['science'])

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{PhDmath,bm}")

seed = None
# seed = 12345

seed_everything(seed)  # PyTorch Lightning seeding


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
# n_x = 32

# var_y_x_const = 1 / (n_x-1)
var_y_x_const = 1/5
# var_y_x_const = 1/50

alpha_y_x_d = (1-var_y_x_const) / (np.float64(var_y_x_const) - 1/(n_x-1))
alpha_y_x_beta = 1/var_y_x_const - 1


# True model
shape_x = ()
# shape_x = (2,)

# w_model = [.5]
w_model = [0, 1]

# nonlinear_model = funcs.make_sin_orig(shape_x)
nonlinear_model = funcs.make_rand_discrete(n_x, rng=seed)

supp_x = box_grid(np.broadcast_to([0, 1], (*shape_x, 2)), n_x, endpoint=True)
model_x = rand_elements.Finite(supp_x, p=np.full(math.prod(shape_x)*(n_x,), n_x**-math.prod(shape_x)))
# model = rand_models.DataConditional(poly_mean_to_models(n_x, alpha_y_x_d, w_model), model_x)
model = rand_models.DataConditional(func_mean_to_models(n_x, alpha_y_x_d, nonlinear_model), model_x)

# model_x = rand_elements.Uniform(np.broadcast_to([0, 1], (*shape_x, 2)))
# # model = rand_models.BetaLinear(weights=w_model, basis_y_x=None, alpha_y_x=alpha_y_x_beta, model_x=model_x)
# model = rand_models.BetaLinear(weights=[1], basis_y_x=[nonlinear_model], alpha_y_x=alpha_y_x_beta, model_x=model_x)

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

w_prior = [.5, 0]
# w_prior = [0, 1]


# Dirichlet learner
proc_funcs = []

prior_mean = rand_models.DataConditional(poly_mean_to_models(n_x, alpha_y_x_d, w_prior), model_x)
# _func = lambda x: .5*(1-np.sin(2*np.pi*x))
# prior_mean = rand_models.DataConditional(func_mean_to_models(n_x, alpha_y_x_d, _func), model_x)


# n_t = 16
# supp_t = box_grid(model_x.lims, n_t, endpoint=True)
# # _temp = np.ones(model_x.size*(n_t,))
# _temp = prob_disc(model_x.size*(n_t,))
# # prior_mean_x = rand_elements.Finite(supp_t, p=_temp/_temp.sum())
# prior_mean_x = rand_elements.DataEmpirical(supp_t, counts=_temp, space=model_x.space)
# proc_funcs.append(discretizer(supp_t.reshape(-1, *model_x.shape)))
#
# prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x_beta, model_x=prior_mean_x)


dir_predictor = BayesRegressor(bayes_models.Dirichlet(prior_mean, alpha_0=10),
                               space=model.space, proc_funcs=proc_funcs, name=r'$\Dir$')

# dir_params = {}
# dir_params = {'alpha_0': [10, 1000]}
# dir_params = {'alpha_0': [.001]}
# dir_params = {'alpha_0': [20]}
# dir_params = {'alpha_0': [.01, 100]}
dir_params = {'alpha_0': [1e-6, 1e6]}
# dir_params = {'alpha_0': [40, 400, 4000]}
# dir_params = {'alpha_0': 1e-6 + np.linspace(0, 20, 100)}
# dir_params = {'alpha_0': np.logspace(-0., 5., 60)}
# dir_params = {'alpha_0': np.logspace(-3., 3., 60)}

if do_bayes:  # add true bayes model concentration
    if model_eval.alpha_0 not in dir_params['alpha_0']:
        dir_params['alpha_0'] = np.sort(np.concatenate((dir_params['alpha_0'], [model_eval.alpha_0])))


# Normal learner

# w_prior_norm = w_prior
# w_prior_norm = [.5, *(0 for __ in range(n_x-1))]
w_prior_norm = [.5, *(0 for __ in range(4))]
basis_y_x = None

# def make_delta_func(val):
#     return lambda x: np.where(x == val, 1, 0)
# w_prior_norm = [.5 for __ in supp_x]
# basis_y_x = [make_delta_func(val) for val in supp_x]

# def make_square_func(val):
#     delta = 0.5 / (supp_t.size-1)
#     return lambda x: np.where((x >= val-delta) & (x < val+delta), 1, 0)
# w_prior_norm = [.5 for __ in supp_t]
# basis_y_x = [make_square_func(val) for val in supp_t]

norm_predictor = BayesRegressor(bayes_models.NormalLinear(prior_mean=w_prior_norm, prior_cov=.1, basis_y_x=basis_y_x,
                                                          cov_y_x=.1, model_x=model_x, allow_singular=True),
                                space=model.space, name=r'$\Ncal$')

# norm_params = {}
# norm_params = {'prior_cov': [.1, .001]}
norm_params = {'prior_cov': [1e6]}
# norm_params = {'prior_cov': [.1 / (.001 / n_x)]}
# norm_params = {'prior_cov': [.1 / (20 / n_t)]}
# norm_params = {'prior_cov': [100, .001]}
# norm_params = {'prior_cov': np.logspace(-7., 3., 60)}


#%% Scikit-Learn
# skl_estimator, _name = LinearRegression(), 'LR'
# skl_estimator, _name = SGDRegressor(max_iter=1000, tol=None), 'SGD'
# skl_estimator, _name = GaussianProcessRegressor(), 'GP'

# _solver_kwargs = {'solver': 'sgd', 'learning_rate': 'adaptive', 'learning_rate_init': 1e-1, 'n_iter_no_change': 20}
_solver_kwargs = {'solver': 'adam', 'learning_rate_init': 1e-3, 'n_iter_no_change': 200}
# _solver_kwargs = {'solver': 'lbfgs', }
skl_estimator, skl_name = MLPRegressor(hidden_layer_sizes=[1000, 200, 100], alpha=0, verbose=True,
                                       max_iter=5000, tol=1e-8, **_solver_kwargs), 'MLP'

# TODO: try Adaboost, RandomForest, GP, BayesianRidge, KNeighbors, SVR

# skl_estimator = Pipeline([('scaler', StandardScaler()), ('regressor', skl_estimator)])
skl_predictor = SKLWrapper(skl_estimator, space=model.space, name=skl_name)


#%% PyTorch
weight_decays = [0.]
# weight_decays = [0.001]
# weight_decays = [0., 0.001]

lit_predictors = []
for weight_decay in weight_decays:
    layer_sizes = [500, 500, 500]
    optim_params = {'lr': 1e-3, 'weight_decay': weight_decay}

    lit_name = f"Lit MLP {'-'.join(map(str, layer_sizes))}, {optim_params['weight_decay']} reg."

    trainer_params = {
        'max_epochs': 10000,
        'callbacks': pl.callbacks.EarlyStopping('train_loss', min_delta=1e-4, patience=500,
                                                check_on_train_epoch_end=True),
        'checkpoint_callback': False,
        'logger': pl_loggers.TensorBoardLogger('logs/', name=lit_name),
        'weights_summary': None,
        'gpus': torch.cuda.device_count(),
    }

    lit_model = LitMLP([math.prod(shape_x), *layer_sizes], optim_params=optim_params)
    lit_predictor = LitWrapper(lit_model, model.space, trainer_params, name=lit_name)
    lit_predictors.append(lit_predictor)


#%% Results

n_train = 5
# n_train = [1, 4, 40, 400]
# n_train = [20, 40, 200, 400, 2000]
# n_train = 2**np.arange(11)
# n_train = [0, 400, 4000]
# n_train = np.arange(0, 55, 5)
# n_train = np.arange(0, 4500, 500)
# n_train = np.concatenate((np.arange(0, 250, 50), np.arange(200, 4050, 50)))

n_test = 1000

n_mc = 5


temp = [
    (opt_predictor, None),
    (dir_predictor, dir_params),
    # *(zip(dir_predictors, dir_params_full)),
    (norm_predictor, norm_params),
    # (skl_predictor, None),
    *((predictor, None) for predictor in lit_predictors),
]
predictors, params = zip(*temp)


# file = None
file = 'docs/temp/results.md'

if file is not None:
    file = Path(file).open('a')


# FIXME: need clipping to Box space for fair comparison?? Clipping to Finite is nonsensical?

# y_stats_full, loss_full = results.predictor_compare(predictors, model_eval, params, n_train, n_test, n_mc,
#                                                     stats=('mean', 'std'), plot_stats=True, print_loss=True,
#                                                     verbose=True, img_path='images/temp/', file=file)

# y_stats_full, loss_full = results.predictor_compare(predictors, model_eval, params, n_train, n_test, n_mc,
#                                                     plot_loss=True, print_loss=True,
#                                                     verbose=True, img_path='images/temp/', file=file)

results.plot_fit_compare(predictors, model.rvs(n_train), model.rvs(n_test), params,
                         img_path='images/temp/', file=file)

# TODO: include `plot_fit_compare` figs in dissertation!?

if file is not None:
    file.close()


#%% Deprecated

# plot_predict_stats_compare(predictors, model_eval, params, n_train, n_mc=100, x=None, do_std=True, verbose=True)
# plot_risk_eval_sim_compare(predictors, model_eval, params, n_train, n_test=100, n_mc=100, verbose=True)

# risk_eval_sim_compare(predictors, model_eval, params, n_train, n_test=100, n_mc=10, verbose=True, print_loss=True)


# # Save image and Figure
# # TODO: move to plotting functions, make path arg
# image_path = Path('./images/temp/')
#
# fig = plt.gcf()
# fig.savefig(image_path.joinpath(f"{NOW_STR}.png"))
# with open(image_path.joinpath(f"{NOW_STR}.mpl"), 'wb') as fid:
#     pickle.dump(fig, fid)


# model_x = rand_elements.Finite([0, .5], p=None)
# model = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.5, .5)], model_x)
# prior_mean = rand_models.DataConditional([rand_elements.Finite([0, .5], [p, 1 - p]) for p in (.9, .9)], model_x)


# _name = r'$\mathrm{Dir}$'
# if len(proc_funcs) > 0:
#     _card = str(n_t)
#     if model_x.size > 1:
#         _card += f"^{model_x.size}"
#     _name += r', $|\mathcal{T}| = __card__$'.replace('__card__', _card)


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
#     prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x_beta,
#                                         model_x=prior_mean_x)
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
