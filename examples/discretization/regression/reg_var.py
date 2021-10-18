# TODO: CLEANUP


from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
import torch
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything

from stats_learn.util.base import get_now
from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors.base import ModelRegressor, BayesRegressor
from stats_learn import results
from stats_learn.util.math import prob_disc
from stats_learn.util.data_processing import make_discretizer
from stats_learn.util.data_processing import make_clipper
from stats_learn.predictors.torch import LitMLP, LitWrapper, reset_weights


plt.style.use('../../../images/style.mplstyle')

# seed = None
seed = 12345

if seed is not None:
    seed_everything(seed)  # PyTorch-Lightning seeding


# log_path = None
# img_path = None

# TODO: remove path stuff and image names below before release
base_path = __file__[__file__.rfind('/')+1:].removesuffix('.py') + '_temp/'
log_path = base_path + 'log.md'
img_dir = base_path + f'images/{get_now()}/'


# %% Model and optimal predictor
freq = 2

# def clairvoyant_func(x):
#     # y = np.sin(2*np.pi*freq*x)
#     # y = np.where(y > 0, .75, .25)
#     # return y
#     a = .35
#     return .5 + a * np.sin(2 * np.pi * freq * x)


# def clairvoyant_func(x):  # linear modulated square wave
#     y = np.sin(2*np.pi*freq*x)
#     y = np.where(y > 0, .5, -.5)
#     return .5 + y * (.2 + .4 * x)


# def clairvoyant_func(x):
#     y = 2 * freq * x % 1
#     return y

def clairvoyant_func(x):
    y = np.sin(2 * np.pi * freq * x)
    return .5 + np.where(y > 0, .3, -.3) - .3*y


var_y_x_const = 1/2


alpha_y_x = 1/var_y_x_const - 1
model_x = rand_elements.Uniform([0, 1])
model = rand_models.BetaLinear(weights=[1], basis_y_x=[clairvoyant_func], alpha_y_x=alpha_y_x, model_x=model_x)

opt_predictor = ModelRegressor(model, name=r'$f_{\Theta}(\theta)$')


# %% Learners

# Dirichlet
def prior_func(x):
    # return .5 + .35*np.sin(2*np.pi*freq*x)
    y = np.sin(2*np.pi*freq*x)
    a = .25
    return np.where(y > 0, .5 + a, .5 - a)


# n_t_iter = [4, 128, 4096]
# n_t_iter = [4, 8, 16, 32, 64, 128, 4096]
# n_t_iter = [32, 64, 128, 256]
# n_t_iter = [8, 16, 32, 64, 128]
# n_t_iter = [8, 32, 128]
n_t_iter = [16]
# n_t_iter = [16, 32, 64]
# n_t_iter = 2 ** np.arange(1, 8)


# alpha_0_norm = 5
# dir_params_full = [None for __ in n_t_iter]
# alpha_0_norm_iter = [6.25]
alpha_0_norm_iter = [1e-6, 4.5]
# alpha_0_norm_iter = [4.5]
# alpha_0_norm_iter = 6.25 * np.array([1e-3, 1])
# alpha_0_norm_iter = [4]
# alpha_0_norm_iter = [.005, 5]
dir_params_full = [None for __ in range(len(n_t_iter) * len(alpha_0_norm_iter))]
dir_predictors = []
for n_t in n_t_iter:
    for alpha_0_norm in alpha_0_norm_iter:
        # supp_t = np.linspace(*model_x.lims, n_t)
        # counts = prob_disc(supp_t.shape)
        supp_t = np.linspace(*model_x.lims, n_t, endpoint=False) + .5 / n_t  # FIXME: cheap?
        counts = np.ones_like(supp_t)

        prior_mean_x = rand_elements.DataEmpirical(supp_t, counts, space=model_x.space)
        prior_mean = rand_models.BetaLinear(weights=[1], basis_y_x=[prior_func], alpha_y_x=alpha_y_x,
                                            model_x=prior_mean_x)

        dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=alpha_0_norm * n_t)

        # FIXME
        name_ = r'$\mathrm{Dir}$, $|\mathcal{T}| = ' + f"{n_t}$" + \
                r", $\alpha_0 / |\mathcal{T}| = " + f"{alpha_0_norm}$"

        dir_predictor = BayesRegressor(dir_model, space=model.space, proc_funcs=[make_discretizer(supp_t)], name=name_)

        dir_predictors.append(dir_predictor)


# scale_alpha = True  # interpret `alpha_0` parameter as normalized w.r.t. discretization cardinality
# # scale_alpha = False
#
# # dir_params = {'alpha_0': [500]}
# dir_params = {'alpha_0': np.logspace(-3, 3, 60)}
#
# dir_params_full = [deepcopy(dir_params) for __ in n_t_iter]
# dir_predictors = []
# for n_t, _params in zip(n_t_iter, dir_params_full):
#     # supp_t = np.linspace(*model_x.lims, n_t)
#     # counts = prob_disc(supp_t.shape)
#     supp_t = np.linspace(*model_x.lims, n_t, endpoint=False) + .5 / n_t  # FIXME: cheap?
#     counts = np.ones_like(supp_t)
#
#     prior_mean_x = rand_elements.DataEmpirical(supp_t, counts, space=model_x.space)
#     prior_mean = rand_models.BetaLinear(weights=[1], basis_y_x=[prior_func], alpha_y_x=alpha_y_x,
#                                         model_x=prior_mean_x)
#     dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)
#
#     name_ = r'$\mathrm{Dir}$, $|\mathcal{T}| = ' + f"{n_t}$"
#
#     dir_predictor = BayesRegressor(dir_model, space=model.space, proc_funcs=[make_discretizer(supp_t)], name=name_)
#
#     dir_predictors.append(dir_predictor)
#
#     if scale_alpha and _params is not None:
#         _params['alpha_0'] *= n_t


# PyTorch
weight_decays = [0., 3e-3]

proc_funcs = {'pre': [], 'post': [make_clipper(model_x.lims)]}

lit_predictors = []
for weight_decay in weight_decays:
    layer_sizes = [500, 500, 500, 500]
    optim_params = {'lr': 1e-3, 'weight_decay': weight_decay}

    logger_name = f"MLP {'-'.join(map(str, layer_sizes))}, lambda {weight_decay}"
    lit_name = r"$\mathrm{MLP}$, " + fr"$\lambda = {weight_decay}$"

    trainer_params = {
        'max_epochs': 100000,
        'callbacks': EarlyStopping('train_loss', min_delta=1e-3, patience=10000, check_on_train_epoch_end=True),
        'checkpoint_callback': False,
        # 'logger': False,
        'logger': pl_loggers.TensorBoardLogger(base_path + 'logs/', name=logger_name),
        'weights_summary': None,
        'gpus': torch.cuda.device_count(),
    }

    lit_model = LitMLP([model.size['x'], *layer_sizes, 1], optim_params=optim_params)

    def reset_func(model_):
        model_.apply(reset_weights)
        # with torch.no_grad():
        #     model_.model[-1].bias.fill_(.5)  # FIXME: use the .5 init??

    lit_predictor = LitWrapper(lit_model, model.space, trainer_params, reset_func, proc_funcs, name=lit_name)
    lit_predictors.append(lit_predictor)


#
temp = [
    (opt_predictor, None),
    # *zip(dir_predictors, dir_params_full),
    *((predictor, None) for predictor in lit_predictors),
]
predictors, params = zip(*temp)


# %% Results
n_test = 1000
n_mc = 50


# # Sample regressor realizations
# n_train = 128
# d = model.rvs(n_train + n_test, rng=seed)
# d_train, d_test = np.split(d, [n_train])
# x_plt = np.linspace(0, 1, 10000)
#
# img_path = img_dir + 'fit.png'
# loss_full = results.plot_fit_compare(predictors, d_train, d_test, params, x_plt, verbose=True,
#                                      log_path=log_path, img_path=img_path)

# Prediction mean/variance, comparative
n_train = 128

img_path = img_dir + 'predict_T.png'
y_stats_full, loss_full = results.assess_compare(predictors, model, params, n_train, n_test, n_mc,
                                                 stats=('mean', 'std'), verbose=True,
                                                 plot_stats=True, print_loss=True,
                                                 log_path=log_path, img_path=img_path, rng=seed)

# # Dirichlet-based prediction mean/variance, varying N
# n_train = [0, 400, 4000]
#
# img_path = img_dir + f'predict_N_T{n_t_iter[0]}.png'
# y_stats_full, loss_full = dir_predictors[0].assess(model, dir_params_full[0], n_train, n_test, n_mc,
#                                                    stats=('mean', 'std'),
#                                                    verbose=True, plot_stats=True, print_loss=True,
#                                                    log_path=log_path, img_path=img_path, rng=seed)

# # Squared-Error vs. training data volume N
# n_train = np.insert(2**np.arange(12), 0, 0)
#
# img_path = img_dir + 'risk_N_leg_T.png'
# y_stats_full, loss_full = results.assess_compare(predictors, model, params, n_train, n_test, n_mc, verbose=True,
#                                                  plot_loss=True, print_loss=True, log_path=log_path,
#                                                  img_path=img_path, rng=seed)

# # Squared-Error vs. prior localization alpha_0
# n_train = 128
#
# img_path = img_dir + 'risk_a0norm_leg_T.png'
# y_stats_full, loss_full = results.assess_compare(dir_predictors, model, dir_params_full, n_train, n_test, n_mc,
#                                                  verbose=True, plot_loss=True, print_loss=True, log_path=log_path,
#                                                  img_path=img_path, rng=seed)
#
# ax = plt.gca()
# if ax.get_xlabel() == r'$\alpha_0$':  # scale alpha axis, find localization minimum
#     ax.set_xscale('log')
#     lines = ax.get_lines()
#     for line in lines:
#         x_, y_ = line.get_data()
#         if scale_alpha:
#             label = line.get_label()
#             _n_t = int(label[label.find('=')+1:-1])
#             x_ /= _n_t
#             line.set_data(x_, y_)
#
#     if scale_alpha:
#         ax.set_xlabel(r'$\alpha_0 / |\mathcal{T}|$')
#         _vals = dir_params['alpha_0']
#         ax.set_xlim((min(_vals), max(_vals)))


# do_argmin = False
# # do_argmin = True
# ax = plt.gca()
# if ax.get_xlabel() == r'$\alpha_0$':  # scale alpha axis, find localization minimum
#     ax.set_xscale('log')
#     lines = ax.get_lines()
#     for line in lines:
#         x_, y_ = line.get_data()
#         if scale_alpha:
#             label = line.get_label()
#             _n_t = int(label[label.find('=')+1:-1])
#             x_ /= _n_t
#             line.set_data(x_, y_)
#
#         if do_argmin:
#             idx = y_.argmin()
#             x_i, y_i = x_[idx], y_[idx]
#             ax.plot(x_i, y_i, marker='.', markersize=8, color=line.get_color())
#     if scale_alpha:
#         ax.set_xlabel(r'$\alpha_0 / |\mathcal{T}|$')
#         _vals = dir_params['alpha_0']
#         ax.set_xlim((min(_vals), max(_vals)))


# n_train = [16, 128, 512]
# # n_train = 400
# results.plot_risk_disc(dir_predictors, model, dir_params_full, n_train, n_test, n_mc, verbose=True, ax=None)
# plt.xscale('log', base=2)
