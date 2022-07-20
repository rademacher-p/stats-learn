import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from stats_learn import bayes, random
from stats_learn.predictors import BayesRegressor
from stats_learn.util import get_now

# # Input
parser = argparse.ArgumentParser(description="Example: Bayesian squared-error on a discrete domain")
parser.add_argument(
    "sims",
    nargs="*",
    choices=["risk_bayes_N_leg_a0", "risk_bayes_a0_leg_N"],
    help="Simulations to run",
)
parser.add_argument("-m", "--mc", type=int, default=1, help="Number of Monte Carlo iterations")
parser.add_argument("-l", "--log_path", default=None, help="Path to log file")
parser.add_argument("-i", "--save_img", action="store_true", help="Save images to log")
parser.add_argument("--style", default=None, help="Path to Matplotlib style")
parser.add_argument("--seed", type=int, default=None, help="RNG seed")

args = parser.parse_args()

sim_names = args.sims
n_mc = args.mc

log_path = args.log_path
if log_path is not None:
    log_path = Path(log_path)

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
var_y_x_const = 1 / 5
w_model = [0.5]

model_x = random.elements.FiniteGeneric.from_grid([0, 1], n_x, p=None)

alpha_y_x = (1 - var_y_x_const) / (np.float64(var_y_x_const) - 1 / (n_y - 1))
prior_mean = random.models.DataConditional.from_mean_poly_emp(alpha_y_x, n_x, w_model, model_x)
model = bayes.models.Dirichlet(prior_mean, alpha_0=4e2)


# # Dirichlet Learner
dir_model = bayes.models.Dirichlet(prior_mean, alpha_0=10)
dir_predictor = BayesRegressor(dir_model, name=r"$\mathrm{Dir}$")


# # Results
n_test = 100

# Bayes Squared-Error vs. N
if "risk_bayes_N_leg_a0" in sim_names:
    n_train = np.linspace(0, 4000, 81, dtype=int)
    dir_params = {"alpha_0": [40, 400, 4000]}

    dir_predictor.model_assess(
        model,
        dir_params,
        n_train,
        n_test,
        n_mc,
        verbose=True,
        plot_loss=True,
        print_loss=False,
        log_path=log_path,
        img_path=get_img_path("risk_bayes_N_leg_a0"),
        rng=seed,
    )

# Bayes Squared-Error vs. prior localization alpha_0
if "risk_bayes_a0_leg_N" in sim_names:
    n_train = [0, 100, 200, 400, 800]
    dir_params = {"alpha_0": np.sort(np.concatenate((np.logspace(-0.0, 5.0, 60), [model.alpha_0])))}

    dir_predictor.model_assess(
        model,
        dir_params,
        n_train,
        n_test,
        n_mc,
        verbose=True,
        plot_loss=True,
        print_loss=False,
        log_path=log_path,
        img_path=get_img_path("risk_bayes_a0_leg_N"),
        rng=seed,
    )

    plt.gca().set_xscale("log")

plt.show()
