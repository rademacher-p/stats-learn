import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from stats_learn import random, bayes
from stats_learn.predictors.base import BayesRegressor
from stats_learn.util import get_now


# # Input
parser = argparse.ArgumentParser(description='Example: Bayesian risk for Dirichlet regressor on a continuous domain')
parser.add_argument('sims', nargs='*', choices=['risk_bayes_N_leg_a0'], help=f'Simulations to run')
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
var_y_x_const = 1 / 5
w_model = [.5]

model_x = random.elements.Uniform([0, 1])

alpha_y_x = 1 / var_y_x_const - 1
prior_mean = random.models.BetaLinear(weights=w_model, alpha_y_x=alpha_y_x, model_x=model_x)
model = bayes.models.Dirichlet(prior_mean, alpha_0=4e2)


# # Dirichlet Learner
dir_model = bayes.models.Dirichlet(prior_mean, alpha_0=10)
dir_predictor = BayesRegressor(dir_model, name=r'$\mathrm{Dir}$')


# # Results
n_test = 100

# Bayes Squared-Error vs. N
if 'risk_bayes_N_leg_a0' in sim_names:
    n_train = np.arange(0, 4050, 50)
    dir_params = {'alpha_0': [40, 400, 4000]}

    dir_predictor.assess(model, dir_params, n_train, n_test, n_mc, verbose=True, plot_loss=True, print_loss=False,
                         log_path=log_path, img_path=get_img_path('risk_bayes_N_leg_a0.png'), rng=seed)
