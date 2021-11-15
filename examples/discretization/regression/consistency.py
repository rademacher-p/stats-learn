import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from stats_learn import results
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors.base import ModelRegressor, BayesRegressor
from stats_learn.preprocessing import make_discretizer, prob_disc
from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.util import get_now


# # Input
parser = argparse.ArgumentParser(description='Example: consistent discretized regressor on a continuous domain')
parser.add_argument('-m', '--mc', type=int, default=1, help='Number of Monte Carlo iterations')
parser.add_argument('-l', '--log_path', type=str, default=None, help='Path to log file')
parser.add_argument('-i', '--save_img', action="store_true", help='Save images to log')
parser.add_argument('--style', type=str, default=None, help='Path to .mplstyle Matplotlib style')
parser.add_argument('--seed', type=int, default=None, help='RNG seed')

args = parser.parse_args()

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


def clairvoyant_func(x):
    return 1 / (2 + np.sin(2 * np.pi * x))


model_x = rand_elements.Uniform([0, 1])

alpha_y_x = 1 / var_y_x_const - 1
model = rand_models.BetaLinear(weights=[1], basis_y_x=[clairvoyant_func], alpha_y_x=alpha_y_x, model_x=model_x)

opt_predictor = ModelRegressor(model, name=r'$f^*(\theta)$')


# # Learners
w_prior = [.5, 0]

# Dirichlet
n_t_iter = [4, 128, 4096]

alpha_0_norm = .1
dir_params_full = [None for __ in n_t_iter]
dir_predictors = []
for n_t in n_t_iter:
    values_t = np.linspace(*model_x.lims, n_t)
    counts = prob_disc(values_t.shape)

    prior_mean_x = rand_elements.DataEmpirical(values_t, counts, space=model_x.space)
    prior_mean = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x,
                                        model_x=prior_mean_x)

    dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=alpha_0_norm * n_t)

    name_ = r'$\mathrm{Dir}$, $|\mathcal{T}| = ' + f"{n_t}$" + r", $\alpha_0 / |\mathcal{T}| = " + f"{alpha_0_norm}$"
    dir_predictor = BayesRegressor(dir_model, space=model.space, proc_funcs=[make_discretizer(values_t)], name=name_)

    dir_predictors.append(dir_predictor)


scale_alpha = True  # interpret `alpha_0` parameter as normalized w.r.t. discretization cardinality


def make_normalized(n_t_iter_, dir_params_):
    dir_params_full_ = [dir_params_.copy() for __ in n_t_iter_]
    dir_predictors_ = []
    for n_t_, params_ in zip(n_t_iter_, dir_params_full_):
        values_t_ = np.linspace(*model_x.lims, n_t_)

        prior_mean_x_ = rand_elements.DataEmpirical(values_t_, counts=prob_disc(values_t_.shape), space=model_x.space)
        prior_mean_ = rand_models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x,
                                             model_x=prior_mean_x_)

        dir_model_ = bayes_models.Dirichlet(prior_mean_, alpha_0=10)
        dir_predictor_ = BayesRegressor(dir_model_, space=model.space, proc_funcs=[make_discretizer(values_t_)],
                                        name=r'$\mathrm{Dir}$, $|\mathcal{T}| = ' + f"{n_t_}$")

        dir_predictors_.append(dir_predictor_)

        if scale_alpha and params_ is not None:
            params_['alpha_0'] = n_t_ * np.array(params_['alpha_0'])

    return dir_predictors_, dir_params_full_


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


# # Results
n_test = 1000

# Sample regressor realizations
n_train = 30
d = model.sample(n_train + n_test, rng=seed)
d_train, d_test = np.split(d, [n_train])
x_plt = np.linspace(0, 1, 10000)

results.assess_single_compare(predictors, d_train, d_test, params, x_plt, verbose=True, log_path=log_path,
                              img_path=get_img_path('fit.png'))

# Prediction mean/variance, comparative
n_train = 400

results.assess_compare(predictors, model, params, n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                       plot_stats=True, print_loss=True, log_path=log_path, img_path=get_img_path('predict_T.png'),
                       rng=seed)

# Dirichlet-based prediction mean/variance, varying N
n_train = [0, 400, 4000]
_t = 4
_alpha_0_norm = 250
dir_params = {'alpha_0': [_alpha_0_norm * _t]}
idx = n_t_iter.index(_t)
_title = r'$\mathrm{Dir}$, $|\mathcal{T}| = ' + f"{_t}$" + r", $\alpha_0 / |\mathcal{T}| = " + f"{_alpha_0_norm}$"

dir_predictors[idx].assess(model, dir_params, n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                           plot_stats=True, print_loss=True, log_path=log_path,
                           img_path=get_img_path(f'predict_N_T{_t}.png'), rng=seed)

plt.gca().set(title=_title)

# Squared-Error vs. training data volume N
n_train = np.arange(0, 4050, 50)

results.assess_compare(predictors, model, params, n_train, n_test, n_mc, verbose=True, plot_loss=True, print_loss=True,
                       log_path=log_path, img_path=get_img_path('risk_N_leg_T.png'), rng=seed)

# Squared-Error vs. prior localization alpha_0
n_train = 4
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
n_train = [0, 4, 40, 400]
dir_predictors, dir_params_full = make_normalized(2 ** np.arange(1, 14), {'alpha_0': [.1]})

results.plot_risk_disc(dir_predictors, model, dir_params_full, n_train, n_test, n_mc, verbose=True, ax=None)

plt.xscale('log', base=2)

# Squared-Error vs. discretization |T|, various alpha_0
n_train = 400
dir_predictors, dir_params_full = make_normalized(2 ** np.arange(1, 14), {'alpha_0': [.1, 10]})

results.plot_risk_disc(dir_predictors, model, dir_params_full, n_train, n_test, n_mc, verbose=True, ax=None)

plt.xscale('log', base=2)
