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


#%% Model and optimal predictor
n_x = n_y = 128
n_x = n_y = 32  # FIXME

freq = 2

# def clairvoyant_func(x):
#     # y = np.sin(2*np.pi*freq*x)
#     # y = np.where(y > 0, .75, .25)
#     # return y
#     return .5 + .35 * np.sin(2 * np.pi * freq * x)


def clairvoyant_func(x):
    y = np.sin(2 * np.pi * freq * x)
    return .5 + np.where(y > 0, .3, -.3) - .3*y


# var_y_x_const = 1/5
var_y_x_const = 1/2
# var_y_x_const = .8  # FIXME


supp_x = np.linspace(0, 1, n_x)
model_x = rand_elements.Finite(supp_x, p=None)

alpha_y_x = (1-var_y_x_const) / (np.float64(var_y_x_const) - 1/(n_y-1))
model = rand_models.DataConditional.from_func_mean(n_y, alpha_y_x, clairvoyant_func, model_x)

opt_predictor = ModelRegressor(model, name=r'$f_{\Theta}(\theta)$')


#%% Learners

# Dirichlet
# def prior_func(x):
#     # return .5 + .35*np.sin(2*np.pi*freq*x)
#     y = np.sin(2 * np.pi * freq * x)
#     y = np.where(y > 0, .75, .25)
#     return y

def prior_func(x):
    # return .5 + .35*np.sin(2*np.pi*freq*x)
    y = np.sin(2*np.pi*freq*x)
    a = .25
    # a = .15
    return np.where(y > 0, .5 + a, .5 - a)


prior_mean = rand_models.DataConditional.from_func_mean(n_y, alpha_y_x, prior_func, model_x)
dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)

dir_predictor = BayesRegressor(dir_model, space=model.space, name=r'$\mathrm{Dir}$')

# dir_params = {'alpha_0': [8e-5, 800]}
# dir_params = {'alpha_0': [5e-5, 500]}
# dir_params = {'alpha_0': [1e-5, 120]}
# dir_params = {'alpha_0': [1e-5, 2e3]}
# dir_params = {'alpha_0': [1e-5, 6e2]}  # 32pt, var_c=.8, a=.15 prior
# dir_params = {'alpha_0': [1e-5, 220]}  # 32pt, var_c=.8, a=.25 prior
dir_params = {'alpha_0': [1e-5, 125]}  # 32pt, var_c=.5, a=.25 prior


# PyTorch
weight_decays = [0., 1e-3]  # controls L2 regularization
# weight_decays = [1e-3]  # FIXME
weight_decays = [0, 1e-3, 3e-3]

proc_funcs = {'pre': [], 'post': [make_clipper([min(supp_x), max(supp_x)])]}

lit_predictors = []
for weight_decay in weight_decays:
    layer_sizes = [500, 500, 500, 500]
    optim_params = {'lr': 1e-3, 'weight_decay': weight_decay}

    logger_name = f"MLP {'-'.join(map(str, layer_sizes))}, lambda {weight_decay}"
    lit_name = r"$\mathrm{MLP}$, " + fr"$\lambda = {weight_decay}$"

    trainer_params = {
        # 'max_epochs': 50000,
        'max_epochs': 100000,
        'callbacks': EarlyStopping('train_loss', min_delta=1e-4, patience=10000, check_on_train_epoch_end=True),
        'checkpoint_callback': False,
        # 'logger': False,
        'logger': pl_loggers.TensorBoardLogger(base_path + 'logs/', name=logger_name),
        'weights_summary': None,
        'gpus': torch.cuda.device_count(),
    }

    lit_model = LitMLP([model.size['x'], *layer_sizes, 1], optim_params=optim_params)

    def reset_func(model_):
        model_.apply(reset_weights)
        with torch.no_grad():
            model_.model[-1].bias.fill_(.5)

    lit_predictor = LitWrapper(lit_model, model.space, trainer_params, reset_func, proc_funcs, name=lit_name)
    lit_predictors.append(lit_predictor)


#
temp = [
    (opt_predictor, None),
    (dir_predictor, dir_params),
    *((predictor, None) for predictor in lit_predictors),
]
predictors, params = zip(*temp)


#%% Results
n_test = 1000
n_mc = 5


# # Sample regressor realizations
# n_train = 128
#
# d = model.rvs(n_train + n_test, rng=seed)
# d_train, d_test = np.split(d, [n_train])
#
# img_path = img_dir + 'fit.png'
# loss_full = results.plot_fit_compare(predictors, d_train, d_test, params, log_path=log_path, img_path=img_path)

# Prediction mean/variance, comparative
n_train = 128

img_path = img_dir + 'predict_full.png'
y_stats_full, loss_full = results.assess_compare(predictors, model, params, n_train, n_test, n_mc,
                                                 stats=('mean', 'std'), verbose=True, plot_stats=True, print_loss=True,
                                                 log_path=log_path, img_path=img_path, rng=seed)

# # Squared-Error vs. training data volume N
# n_train = np.insert(2**np.arange(12), 0, 0)
#
# img_path = img_dir + 'risk_N.png'
# y_stats_full, loss_full = results.assess_compare(predictors, model, params, n_train, n_test, n_mc, verbose=True,
#                                                  plot_loss=True, print_loss=True, log_path=log_path,
#                                                  img_path=img_path, rng=seed)


# # Squared-Error vs. prior localization alpha_0
# n_train = [0, 100, 200, 400]
#
# img_path = img_dir + 'risk_a0_leg_N.png'
# y_stats_full, loss_full = dir_predictor.assess(model, {'alpha_0': np.logspace(0., 5., 60)}, n_train, n_test, n_mc,
#                                                verbose=True, plot_loss=True, print_loss=True, log_path=log_path,
#                                                img_path=img_path, rng=seed)
#
# plt.gca().set_xscale('log')
