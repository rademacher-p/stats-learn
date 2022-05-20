import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from stats_learn import bayes, random, results
from stats_learn.predictors import BayesRegressor, ModelRegressor
from stats_learn.util import get_now

# # Input
parser = argparse.ArgumentParser(description="Example: consistent regressor on a discrete domain")
parser.add_argument(
    "sims",
    nargs="*",
    choices=["fit", "predict_a0", "predict_N", "risk_N_leg_a0", "risk_a0_leg_N"],
    help=f"Simulations to run",
)
parser.add_argument("-m", "--mc", type=int, default=1, help="Number of Monte Carlo iterations")
parser.add_argument("-l", "--log_path", default=None, help="Path to log file")
parser.add_argument("-i", "--save_img", action="store_true", help="Save images to log")
parser.add_argument("--style", default=None, help="Path to .mplstyle Matplotlib style")
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


def clairvoyant_func(x):
    return 1 / (2 + np.sin(2 * np.pi * x))


model_x = random.elements.FiniteGeneric.from_grid([0, 1], n_x, p=None)

alpha_y_x = (1 - var_y_x_const) / (np.float64(var_y_x_const) - 1 / (n_y - 1))
model = random.models.DataConditional.from_mean_emp(alpha_y_x, n_y, clairvoyant_func, model_x)

opt_predictor = ModelRegressor(model, name=r"$f^*(\rho)$")

# # Learners
w_prior = [0.5, 0]

# Dirichlet
prior_mean = random.models.DataConditional.from_mean_poly_emp(alpha_y_x, n_y, w_prior, model_x)
dir_model = bayes.models.Dirichlet(prior_mean, alpha_0=10)

dir_predictor = BayesRegressor(dir_model, space=model.space, name=r"$\mathrm{Dir}$")

dir_params = {"alpha_0": [10, 1000]}

# Normal-prior LR
norm_model = bayes.models.NormalLinear(
    prior_mean=w_prior, prior_cov=0.1, cov_y_x=0.1, model_x=model_x
)
norm_predictor = BayesRegressor(norm_model, space=model.space, name=r"$\mathcal{N}$")

norm_params = {"prior_cov": [0.1, 0.001]}

#
temp = [
    (opt_predictor, None),
    (dir_predictor, dir_params),
    (norm_predictor, norm_params),
]
predictors, params = zip(*temp)

# # Results
n_test = 1000

# Sample regressor realizations
if "fit" in sim_names:
    n_train = 30
    d = model.sample(n_train + n_test, rng=seed)
    d_train, d_test = np.split(d, [n_train])

    results.data_assess(
        predictors,
        d_train,
        d_test,
        params,
        plot_fit=True,
        verbose=True,
        log_path=log_path,
        img_path=get_img_path("fit"),
    )

# Prediction mean/variance, comparative
if "predict_a0" in sim_names:
    n_train = 400

    results.model_assess(
        predictors,
        model,
        params,
        n_train,
        n_test,
        n_mc,
        stats=("mean", "std"),
        verbose=True,
        plot_stats=True,
        print_loss=True,
        log_path=log_path,
        img_path=get_img_path("predict_a0"),
        rng=seed,
    )

# Dirichlet-based prediction mean/variance, varying N
if "predict_N" in sim_names:
    n_train = [0, 800, 4000]

    dir_predictor.model_assess(
        model,
        {"alpha_0": [1000]},
        n_train,
        n_test,
        n_mc,
        stats=("mean", "std"),
        verbose=True,
        plot_stats=True,
        print_loss=True,
        log_path=log_path,
        img_path=get_img_path("predict_N"),
        rng=seed,
    )

# Squared-Error vs. training data volume N
if "risk_N_leg_a0" in sim_names:
    n_train = np.linspace(0, 4000, 81, dtype=int)

    results.model_assess(
        predictors,
        model,
        params,
        n_train,
        n_test,
        n_mc,
        verbose=True,
        plot_loss=True,
        print_loss=True,
        log_path=log_path,
        img_path=get_img_path("risk_N_leg_a0"),
        rng=seed,
    )

# Squared-Error vs. prior localization alpha_0
if "risk_a0_leg_N" in sim_names:
    n_train = [0, 100, 200, 400, 800]

    dir_predictor.model_assess(
        model,
        {"alpha_0": np.logspace(0.0, 5.0, 100)},
        n_train,
        n_test,
        n_mc,
        verbose=True,
        plot_loss=True,
        print_loss=True,
        log_path=log_path,
        img_path=get_img_path("risk_a0_leg_N"),
        rng=seed,
    )

    plt.gca().set_xscale("log")

plt.show()
