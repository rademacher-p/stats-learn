import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning.loggers as pl_loggers
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from stats_learn import random, bayes, results
from stats_learn.predictors.base import ModelRegressor, BayesRegressor
from stats_learn.predictors.torch import LitMLP, LitPredictor
from stats_learn.preprocessing import make_clipper
from stats_learn.preprocessing import make_discretizer
from stats_learn.util import get_now


# # Input
parser = argparse.ArgumentParser(description='Example: discretized regularization against overfitting '
                                             'on a continuous domain')
parser.add_argument('sims', nargs='*', choices=['fit', 'predict', 'risk_N', 'predict_N_T16.png', 'risk_a0norm_leg_T',
                                                'risk_T_leg_N'], help=f'Simulations to run')
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
    def get_img_path(filename):
        return None

if args.style is not None:
    plt.style.use(args.style)

seed = args.seed


# # Model and optimal predictor
var_y_x_const = 1 / 2


def clairvoyant_func(x):
    y = np.sin(2 * np.pi * 2 * x)
    return .5 + np.where(y > 0, .3, -.3) - .3 * y


model_x = random.elements.Uniform([0, 1])

alpha_y_x = 1 / var_y_x_const - 1
model = random.models.BetaLinear(weights=[1], basis_y_x=[clairvoyant_func], alpha_y_x=alpha_y_x, model_x=model_x)

opt_predictor = ModelRegressor(model, name=r'$f^*(\theta)$')


# # Learners

# Dirichlet
def prior_func(x):
    y = np.sin(2 * np.pi * 2 * x)
    a = .25
    return np.where(y > 0, .5 + a, .5 - a)


n_t_iter = [16]
alpha_0_norm_iter = [1e-6, 4.5]

dir_params_full = [None for __ in range(len(n_t_iter) * len(alpha_0_norm_iter))]
dir_predictors = []
for n_t in n_t_iter:
    for alpha_0_norm in alpha_0_norm_iter:
        values_t = np.linspace(*model_x.lims, n_t, endpoint=False) + .5 / n_t
        counts = np.ones_like(values_t)

        prior_mean_x = random.elements.DataEmpirical(values_t, counts, space=model_x.space)
        prior_mean = random.models.BetaLinear(weights=[1], basis_y_x=[prior_func], alpha_y_x=alpha_y_x,
                                              model_x=prior_mean_x)

        dir_model = bayes.models.Dirichlet(prior_mean, alpha_0=alpha_0_norm * n_t)

        name_ = r'$\mathrm{Dir}$, $|\mathcal{T}| = ' + f"{n_t}$" + \
                r", $\alpha_0 / |\mathcal{T}| = " + f"{alpha_0_norm}$"
        dir_predictor = BayesRegressor(dir_model, space=model.space, proc_funcs=[make_discretizer(values_t)],
                                       name=name_)

        dir_predictors.append(dir_predictor)


scale_alpha = True  # interpret `alpha_0` parameter as normalized w.r.t. discretization cardinality


def make_normalized(n_t_iter_, dir_params_):
    dir_params_full_ = [dir_params_.copy() for __ in n_t_iter_]
    dir_predictors_ = []
    for n_t_, params_ in zip(n_t_iter_, dir_params_full_):
        values_t_ = np.linspace(*model_x.lims, n_t_, endpoint=False) + .5 / n_t_

        prior_mean_x_ = random.elements.DataEmpirical(values_t_, counts=np.ones_like(values_t), space=model_x.space)
        prior_mean_ = random.models.BetaLinear(weights=[1], basis_y_x=[prior_func], alpha_y_x=alpha_y_x,
                                               model_x=prior_mean_x_)

        dir_model_ = bayes.models.Dirichlet(prior_mean_, alpha_0=10)
        dir_predictor_ = BayesRegressor(dir_model_, space=model.space, proc_funcs=[make_discretizer(values_t_)],
                                        name=r'$\mathrm{Dir}$, $|\mathcal{T}| = ' + f"{n_t_}$")

        dir_predictors_.append(dir_predictor_)

        if scale_alpha and params_ is not None:
            params_['alpha_0'] = n_t_ * np.array(params_['alpha_0'])

    return dir_predictors_, dir_params_full_


# PyTorch
if seed is not None:
    seed_everything(seed)

weight_decays = [0., 3e-3]

proc_funcs = {'pre': [], 'post': [make_clipper(model_x.lims)]}

lit_predictors = []
for weight_decay in weight_decays:
    layer_sizes = [500, 500, 500, 500]
    optim_params = {'lr': 1e-3, 'weight_decay': weight_decay}

    logger_name = f"MLP {'-'.join(map(str, layer_sizes))}, lambda {weight_decay}"
    lit_name = r"$\mathrm{MLP}$, " + fr"$\lambda = {weight_decay}$"

    if log_path is None:
        logger = False
    else:
        logger_path = str(log_path.parent / 'logs/')
        logger_name = f"MLP {'-'.join(map(str, layer_sizes))}, lambda {weight_decay}"
        logger = pl_loggers.TensorBoardLogger(logger_path, name=logger_name)
    trainer_params = {
        'max_epochs': 100000,
        'callbacks': EarlyStopping('train_loss', min_delta=1e-3, patience=10000, check_on_train_epoch_end=True),
        'checkpoint_callback': False,
        'logger': logger,
        'weights_summary': None,
        'gpus': torch.cuda.device_count(),
    }

    lit_model = LitMLP([model.size['x'], *layer_sizes, 1], optim_params=optim_params)

    lit_predictor = LitPredictor(lit_model, model.space, trainer_params, proc_funcs=(), name=lit_name)
    lit_predictors.append(lit_predictor)

#
temp = [
    (opt_predictor, None),
    *zip(dir_predictors, dir_params_full),
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
    x_plt = np.linspace(0, 1, 10000)

    results.assess_single_compare(predictors, d_train, d_test, params, x_plt, verbose=True, log_path=log_path,
                                  img_path=get_img_path('fit.png'))

# Prediction mean/variance, comparative
if 'predict' in sim_names:
    n_train = 128

    results.assess_compare(predictors, model, params, n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                           plot_stats=True, print_loss=True, log_path=log_path, img_path=get_img_path('predict.png'),
                           rng=seed)

# Squared-Error vs. training data volume N
if 'risk_N' in sim_names:
    n_train = np.insert(2**np.arange(12), 0, 0)

    results.assess_compare(predictors, model, params, n_train, n_test, n_mc, verbose=True, plot_loss=True,
                           print_loss=True, log_path=log_path, img_path=get_img_path('risk_N.png'), rng=seed)

# # Dirichlet-based prediction mean/variance, varying N
if 'predict_N_T16.png' in sim_names:
    n_train = [0, 400, 4000]
    _t = 16

    idx = n_t_iter.index(_t)
    dir_predictors[idx].assess(model, dir_params_full[idx], n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                               plot_stats=True, print_loss=True, log_path=log_path,
                               img_path=get_img_path(f'predict_N_T{_t}.png'), rng=seed)

# Squared-Error vs. prior localization alpha_0
if 'risk_a0norm_leg_T' in sim_names:
    n_train = 128

    dir_params = {'alpha_0': np.logspace(-3, 3, 60)}
    dir_predictors, dir_params_full = make_normalized([2, 4, 8, 16], dir_params)

    results.assess_compare(dir_predictors, model, dir_params_full, n_train, n_test, n_mc, verbose=True, plot_loss=True,
                           print_loss=True, log_path=log_path, img_path=get_img_path('risk_a0norm_leg_T.png'), rng=seed)

    ax = plt.gca()
    if ax.get_xlabel() == r'$\alpha_0$':  # scale alpha axis, find localization minimum
        ax.set_xscale('log')
        lines = ax.get_lines()
        for line in lines:
            x_, y_ = line.get_data()
            if scale_alpha:
                label = line.get_label()
                _n_t = int(label[label.find('=')+1:-1])
                x_ /= _n_t
                line.set_data(x_, y_)

        if scale_alpha:
            ax.set_xlabel(r'$\alpha_0 / |\mathcal{T}|$')
            _vals = dir_params['alpha_0']
            ax.set_xlim((min(_vals), max(_vals)))

# Squared-Error vs. discretization |T|, various N
if 'risk_T_leg_N' in sim_names:
    n_train = [16, 128, 512]

    dir_predictors, dir_params_full = make_normalized(2 ** np.arange(1, 8), {'alpha_0': [4.5]})

    results.plot_risk_disc(dir_predictors, model, dir_params_full, n_train, n_test, n_mc, verbose=True, ax=None)
    plt.xscale('log', base=2)
