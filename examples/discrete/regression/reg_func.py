import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning.loggers as pl_loggers
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from stats_learn import random, bayes, results
from stats_learn.predictors import ModelRegressor, BayesRegressor
from stats_learn.predictors.torch import LitMLP, LitPredictor, reset_weights
from stats_learn.preprocessing import make_clipper
from stats_learn.util import get_now


# # Input
parser = argparse.ArgumentParser(description='Example: regularization against function complexity on a discrete domain')
parser.add_argument('sims', nargs='*', choices=['fit', 'predict', 'risk_N'], help=f'Simulations to run')
parser.add_argument('-m', '--mc', type=int, default=1, help='Number of Monte Carlo iterations')
parser.add_argument('-l', '--log_path', default=None, help='Path to log file')
parser.add_argument('-i', '--save_img', action="store_true", help='Save images to log')
parser.add_argument('--style', default=None, help='Path to .mplstyle Matplotlib style')
parser.add_argument('--seed', type=int, default=None, help='RNG seed')

args = parser.parse_args()

sim_names = args.sims
n_mc = args.mc

log_path = Path(args.log_path)
if log_path is not None and args.save_img:
    img_dir = log_path.parent / f"images/{get_now()}"

    def get_img_path(filename):
        return img_dir / filename
else:
    def get_img_path(_filename):
        return None

if args.style is not None:
    plt.style.use(args.style)

seed = args.seed


# # Model and optimal predictor
n_x = n_y = 128
var_y_x_const = 1 / 125

_rand_vals = dict(zip(np.linspace(0, 1, n_x), np.random.default_rng(seed).random(n_x)))


def clairvoyant_func(x):
    return _rand_vals[x]


model_x = random.elements.FiniteGeneric.from_grid([0, 1], n_x, p=None)

alpha_y_x = (1 - var_y_x_const) / (np.float64(var_y_x_const) - 1 / (n_y - 1))
model = random.models.DataConditional.from_mean_emp(alpha_y_x, n_y, clairvoyant_func, model_x)

opt_predictor = ModelRegressor(model, name=r'$f^*(\theta)$')


# # Learners

# Dirichlet
prior_mean = random.models.DataConditional.from_mean_emp(alpha_y_x, n_y, lambda x: .5, model_x)
dir_model = bayes.models.Dirichlet(prior_mean, alpha_0=10)

dir_predictor = BayesRegressor(dir_model, space=model.space, name=r'$\mathrm{Dir}$')
dir_params = {'alpha_0': [1e-5, 1e5]}

# PyTorch
if seed is not None:
    seed_everything(seed)

weight_decays = [0., 1e-3]  # controls L2 regularization

proc_funcs = {'pre': [], 'post': [make_clipper([np.min(model_x.values), np.max(model_x.values)])]}

lit_predictors = []
for weight_decay in weight_decays:
    layer_sizes = [500, 500, 500, 500]
    optim_params = {'lr': 1e-3, 'weight_decay': weight_decay}

    if log_path is None:
        logger = False
    else:
        logger_path = str(log_path.parent / 'logs')
        logger_name = f"MLP {'-'.join(map(str, layer_sizes))}, lambda {weight_decay}"
        logger = pl_loggers.TensorBoardLogger(logger_path, name=logger_name)
    trainer_params = {
        'max_epochs': 50000,
        'callbacks': EarlyStopping('train_loss', min_delta=1e-6, patience=10000, check_on_train_epoch_end=True),
        'checkpoint_callback': False,
        'logger': logger,
        'weights_summary': None,
        'gpus': torch.cuda.device_count(),
    }

    lit_model = LitMLP([model.size['x'], *layer_sizes, 1], optim_params=optim_params)

    def reset_func(model_):
        model_.apply(reset_weights)
        with torch.no_grad():
            model_.model[-1].bias.fill_(.5)

    lit_name = r"$\mathrm{MLP}$, " + fr"$\lambda = {weight_decay}$"
    lit_predictor = LitPredictor(lit_model, model.space, trainer_params, reset_func, proc_funcs, name=lit_name)
    lit_predictors.append(lit_predictor)

#
temp = [
    (opt_predictor, None),
    (dir_predictor, dir_params),
    *((predictor, None) for predictor in lit_predictors),
]
predictors, params = zip(*temp)


# # Results
n_test = 1000

# Sample regressor realizations
if 'fit' in sim_names:
    n_train = 128
    d = model.sample(n_train + n_test, rng=seed)
    d_train, d_test = np.split(d, [n_train])

    results.data_assess(predictors, d_train, d_test, params, verbose=True, plot_fit=True, log_path=log_path,
                        img_path=get_img_path('fit.png'))

# Prediction mean/variance, comparative
if 'predict' in sim_names:
    n_train = 128

    results.model_assess(predictors, model, params, n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                         plot_stats=True, print_loss=True, log_path=log_path, img_path=get_img_path('predict.png'),
                         rng=seed)

# Squared-Error vs. training data volume N
if 'risk_N' in sim_names:
    n_train = np.insert(np.logspace(0, 10, 11, base=2, dtype=int), 0, 0)

    results.model_assess(predictors, model, params, n_train, n_test, n_mc, verbose=True, plot_loss=True,
                         print_loss=True, log_path=log_path, img_path=get_img_path('risk_N.png'), rng=seed)
