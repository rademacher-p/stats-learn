import math

import numpy as np
from matplotlib import pyplot as plt
import torch
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything

from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors import ModelRegressor, BayesRegressor
from stats_learn.util import funcs, results
from stats_learn.util.data_processing import make_clipper
from stats_learn.util.plotting import box_grid
from stats_learn.util.torch import LitMLP, LitWrapper, reset_weights

plt.rc('text', usetex=True)

seed = 12345

if seed is not None:
    seed_everything(seed)  # PyTorch Lightning seeding


#%% Model and optimal predictor
n_x = 128

shape_x = ()
size_x = math.prod(shape_x)
lims_x = np.broadcast_to([0, 1], (*shape_x, 2))
supp_x = box_grid(lims_x, n_x, endpoint=True)
model_x = rand_elements.Finite(supp_x, p=np.full(size_x*(n_x,), n_x**-size_x))

nonlinear_model = funcs.make_rand_discrete(n_x, rng=seed)
var_y_x_const = 1/125

alpha_y_x = (1-var_y_x_const) / (np.float64(var_y_x_const) - 1/(n_x-1))
model = rand_models.DataConditional.from_func_mean(n_x, alpha_y_x, nonlinear_model, model_x, rng=seed)

opt_predictor = ModelRegressor(model, name=r'$f_{\Theta}(\theta)$')


#%% Dirichlet learner
prior_mean = rand_models.DataConditional.from_func_mean(n_x, alpha_y_x, lambda x: .5, model_x)
dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)

dir_predictor = BayesRegressor(dir_model, space=model.space, name=r'$\mathrm{Dir}$')
dir_params = {'alpha_0': [1e-5, 1e5]}


#%% PyTorch
weight_decays = [0., 1e-3]

proc_funcs = {'pre': [], 'post': [make_clipper(lims_x)]}

lit_predictors = []
for weight_decay in weight_decays:
    layer_sizes = [500, 500, 500, 500]
    optim_params = {'lr': 1e-3, 'weight_decay': weight_decay}

    logger_name = f"MLP {'-'.join(map(str, layer_sizes))}, lambda {weight_decay}"
    lit_name = r"$\mathrm{MLP}$, " + fr"$\lambda = {weight_decay}$"

    trainer_params = {
        'max_epochs': 50000,
        'callbacks': EarlyStopping('train_loss', min_delta=1e-6, patience=10000, check_on_train_epoch_end=True),
        'checkpoint_callback': False,
        'logger': pl_loggers.TensorBoardLogger('logs/learn/', name=logger_name),
        'weights_summary': None,
        'gpus': torch.cuda.device_count(),
    }

    lit_model = LitMLP([size_x, *layer_sizes], optim_params=optim_params)

    def reset_func(model_):
        model_.apply(reset_weights)
        with torch.no_grad():
            model_.model[-1].bias.fill_(.5)

    lit_predictor = LitWrapper(lit_model, model.space, trainer_params, reset_func, proc_funcs, name=lit_name)
    lit_predictors.append(lit_predictor)


#%% Results
n_train = np.concatenate(([0], 2**np.arange(11)))
n_test = 1000
n_mc = 1

temp = [
    (opt_predictor, None),
    (dir_predictor, dir_params),
    # *((predictor, None) for predictor in lit_predictors),
]
predictors, params = zip(*temp)

# y_stats_full, loss_full = results.predictor_compare(predictors, model, params, n_train, n_test, n_mc,
#                                                     stats=('mean', 'std'), plot_stats=True, print_loss=True,
#                                                     verbose=True)

y_stats_full, loss_full = results.predictor_compare(predictors, model, params, n_train, n_test, n_mc,
                                                    plot_loss=True, print_loss=True, verbose=True)


