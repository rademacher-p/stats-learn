import math

# import pickle

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor

import torch
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything

from stats_learn.predictors import ModelRegressor, BayesRegressor
from stats_learn import random, bayes, results
from stats_learn.util import get_now
from stats_learn.preprocessing import make_clipper
from stats_learn.predictors.torch import LitMLP, LitPredictor, reset_weights

np.set_printoptions(precision=3)

plt.style.use("../images/style.mplstyle")
plt.rc("text", usetex=False)
# plt.rc('text.latex', preamble=r"\usepackage{PhDmath}")

# seed = None
seed = 12345

if seed is not None:
    seed_everything(seed)  # PyTorch-Lightning seeding


def make_inv_trig(shape=()):
    def sin_orig(x):
        axis = tuple(range(-len(shape), 0))
        return 1 / (2 + np.sin(2 * np.pi * x.mean(axis)))

    return sin_orig


def make_rand_discrete(n, rng):
    rng = np.random.default_rng(rng)
    _rand_vals = dict(zip(np.linspace(0, 1, n), rng.random(n)))

    def rand_discrete(x):
        return _rand_vals[x]

    return rand_discrete


# # Model and optimal predictor
n_x = 128

# var_y_x_const = 1 / (n_x-1)
var_y_x_const = 1 / 5
# var_y_x_const = 1/125

alpha_y_x_d = (1 - var_y_x_const) / (np.float64(var_y_x_const) - 1 / (n_x - 1))
alpha_y_x_beta = 1 / var_y_x_const - 1


# True model
shape_x = ()
# shape_x = (2,)

size_x = math.prod(shape_x)
lims_x = np.broadcast_to([0, 1], (*shape_x, 2))

w_model = [0.5]

nonlinear_model = make_inv_trig(shape_x)
# nonlinear_model = make_rand_discrete(n_x, rng=seed)


model_x = random.elements.FiniteGeneric.from_grid([0, 1], n_x)
# model = random.models.DataConditional.from_mean_poly_emp(n_x, alpha_y_x_d, w_model, model_x)
model = random.models.DataConditional.from_mean_emp(
    alpha_y_x_d, n_x, nonlinear_model, model_x
)

# model_x = random.elements.Uniform(lims_x)
# # model = random.models.BetaLinear(weights=w_model, basis_y_x=None, alpha_y_x=alpha_y_x_beta, model_x=model_x)
# model = random.models.BetaLinear(weights=[1], basis_y_x=[nonlinear_model], alpha_y_x=alpha_y_x_beta, model_x=model_x)

# model = random.models.NormalLinear(weights=w_model, basis_y_x=None, cov_y_x=.1, model_x=model_x)


# model = bayes.models.Dirichlet(model, alpha_0=4e2)
if isinstance(model, bayes.models.Base):
    opt_predictor = BayesRegressor(model, name=r"$f_\uptheta$")
else:
    opt_predictor = ModelRegressor(model, name=r"$f^*(\rho)$")


# # Bayesian learners

w_prior = [0.5, 0]
# w_prior = [1, 0]
# w_prior = [0, 1]


# Dirichlet learner
proc_funcs = []

prior_mean = random.models.DataConditional.from_mean_poly_emp(
    alpha_y_x_d, n_x, w_prior, model_x
)
# _func = lambda x: .5*(1-np.sin(2*np.pi*x))
# prior_mean = random.models.DataConditional.from_mean_emp(n_x, alpha_y_x_d, _func, model_x)


# n_t = 16
# values_t = box_grid(model_x.lims, n_t, endpoint=True)
# # _temp = np.ones(model_x.size*(n_t,))
# _temp = prob_disc(model_x.size*(n_t,))
# # prior_mean_x = random.elements.FiniteGeneric(values_t, p=_temp/_temp.sum())
# prior_mean_x = random.elements.DataEmpirical(values_t, counts=_temp, space=model_x.space)
# proc_funcs.append(make_discretizer(values_t.reshape(-1, *model_x.shape)))
#
# prior_mean = random.models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x_beta, model_x=prior_mean_x)


dir_model = bayes.models.Dirichlet(prior_mean, alpha_0=10)
dir_predictor = BayesRegressor(
    dir_model, space=model.space, proc_funcs=proc_funcs, name=r"$\mathrm{Dir}$"
)

# dir_params = {}
dir_params = {"alpha_0": [10, 1000]}
# dir_params = {'alpha_0': [.001]}
# dir_params = {'alpha_0': [20]}
# dir_params = {'alpha_0': [.01, 100]}
# dir_params = {'alpha_0': [1e-5, 1e5]}
# dir_params = {'alpha_0': [40, 400, 4000]}
# dir_params = {'alpha_0': [40, 400, 4000]}
# dir_params = {'alpha_0': 1e-6 + np.linspace(0, 20, 100)}
# dir_params = {'alpha_0': np.logspace(-0., 5., 60)}
# dir_params = {'alpha_0': np.logspace(-3., 3., 60)}


if isinstance(model, bayes.models.Dirichlet):  # add true bayes model concentration
    if model.alpha_0 not in dir_params["alpha_0"]:
        dir_params["alpha_0"] = np.sort(
            np.concatenate((dir_params["alpha_0"], [model.alpha_0]))
        )


# Normal learner

w_prior_norm = w_prior
# w_prior_norm = [.5, *(0 for __ in range(n_x-1))]
# w_prior_norm = [.5, *(0 for __ in range(4))]
basis_y_x = None

# def make_delta_func(value):
#     return lambda x: np.where(x == value, 1, 0)
# w_prior_norm = [.5 for __ in model_x.values]
# basis_y_x = [make_delta_func(value) for value in model_x.values]

# def make_square_func(value):
#     delta = 0.5 / (values_t.size-1)
#     return lambda x: np.where((x >= value-delta) & (x < value+delta), 1, 0)
# w_prior_norm = [.5 for __ in values_t]
# basis_y_x = [make_square_func(value) for value in values_t]

# proc_funcs = []
proc_funcs = {"pre": [], "post": [make_clipper(lims_x)]}

norm_model = bayes.models.NormalLinear(
    prior_mean=w_prior_norm,
    prior_cov=0.1,
    basis_y_x=basis_y_x,
    cov_y_x=0.1,
    model_x=model_x,
)
norm_predictor = BayesRegressor(
    norm_model, space=model.space, proc_funcs=proc_funcs, name=r"$\mathcal{N}$"
)

# norm_params = {}
norm_params = {"prior_cov": [0.1, 0.001]}
# norm_params = {'prior_cov': [1e6]}
# norm_params = {'prior_cov': [.1 / (.001 / n_x)]}
# norm_params = {'prior_cov': [.1 / (20 / n_t)]}
# norm_params = {'prior_cov': [100, .001]}
# norm_params = {'prior_cov': np.logspace(-7., 3., 60)}


# # Scikit-Learn
# skl_estimator, _name = LinearRegression(), 'LR'
# skl_estimator, _name = SGDRegressor(max_iter=1000, tol=None), 'SGD'
# skl_estimator, _name = GaussianProcessRegressor(), 'GP'

# _solver_kwargs = {'solver': 'sgd', 'learning_rate': 'adaptive', 'learning_rate_init': 1e-1, 'n_iter_no_change': 20}
_solver_kwargs = {"solver": "adam", "learning_rate_init": 1e-3, "n_iter_no_change": 200}
# _solver_kwargs = {'solver': 'lbfgs', }
skl_estimator, skl_name = (
    MLPRegressor(
        hidden_layer_sizes=[1000, 200, 100],
        alpha=0,
        verbose=True,
        max_iter=5000,
        tol=1e-8,
        **_solver_kwargs,
    ),
    "MLP",
)

# TODO: try Adaboost, RandomForest, GP, BayesianRidge, KNeighbors, SVR

# skl_estimator = Pipeline([('scaler', StandardScaler()), ('regressor', skl_estimator)])
# skl_predictor = SKLPredictor(skl_estimator, space=model.space, name=skl_name)


# # PyTorch
# TODO: add citations to dissertation. PyTorch, Adam weight decay, etc.

# weight_decays = [0.]  # controls L2 regularization
weight_decays = [1e-3]
# weight_decays = [0., 1e-3]

# proc_funcs = []
proc_funcs = {"pre": [], "post": [make_clipper(lims_x)]}

lit_predictors = []
for weight_decay in weight_decays:
    layer_sizes = [500, 500, 500, 500]
    optim_params = {"lr": 1e-3, "weight_decay": weight_decay}

    logger_name = f"MLP {'-'.join(map(str, layer_sizes))}, lambda {weight_decay}"
    # lit_name = r"$\text{{MLP}} {}$, $\lambda = {}$".format('-'.join(map(str, layer_sizes)),
    #                                                      optim_params['weight_decay'])
    lit_name = r"$\mathrm{MLP}$, " + rf"$\lambda = {weight_decay}$"

    trainer_params = {
        "max_epochs": 50000,
        "callbacks": EarlyStopping(
            "train_loss", min_delta=1e-6, patience=10000, check_on_train_epoch_end=True
        ),
        "checkpoint_callback": False,
        "logger": pl_loggers.TensorBoardLogger("temp/logs/", name=logger_name),
        "weights_summary": None,
        "gpus": torch.cuda.device_count(),
    }

    lit_model = LitMLP([model.size["x"], *layer_sizes, 1], optim_params=optim_params)

    def reset_func(model_):
        model_.apply(reset_weights)
        with torch.no_grad():
            # for p in model_.parameters():  # DO NOT USE: breaks gradient descent!
            #     p.data.fill_(0.)
            #     raise Exception
            model_.model[-1].bias.fill_(0.5)

    lit_predictor = LitPredictor(
        lit_model, model.space, trainer_params, reset_func, proc_funcs, name=lit_name
    )

    lit_predictors.append(lit_predictor)


# # Results

n_train = 400
# n_train = [1, 4, 40, 400]
# n_train = [20, 40, 200, 400, 2000]
# n_train = np.insert(2**np.arange(11), 0, 0)
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
    # (norm_predictor, norm_params),
    # (skl_predictor, None),
    *((predictor, None) for predictor in lit_predictors),
]
predictors, params = zip(*temp)


# log_path = None
# img_path = None
log_path = "temp/log.md"
img_path = f"temp/images/{get_now()}"


y_stats_full, loss_full = results.model_assess(
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
    img_path=img_path,
    rng=seed,
)

# y_stats_full, loss_full = results.assess_compare(predictors, model, params, n_train, n_test, n_mc,
#                                                  verbose=True, plot_loss=True, print_loss=True,
#                                                  log_path=log_path, img_path=img_path, rng=seed)

# results.assess_single_compare(predictors, model.sample(n_train), model.sample(n_test), params,
#                          log_path=log_path, img_path=img_path)

# y_stats_full, loss_full = dir_predictor.assess(model, dir_params, n_train, n_test, n_mc,
#                                                verbose=True, plot_loss=True, print_loss=False,
#                                                log_path=log_path, img_path=img_path, rng=seed)


# with open(f'data/temp/{NOW_STR}.pkl', 'wb') as f:
#     pickle.dump(dict(y_stats=y_stats_full, losses=loss_full), f)


# # Deprecated

# plot_predict_stats_compare(predictors, model_eval, params, n_train, n_mc=100, x=None, do_std=True, verbose=True)
# plot_risk_eval_sim_compare(predictors, model_eval, params, n_train, n_test=100, n_mc=100, verbose=True)

# risk_eval_sim_compare(predictors, model_eval, params, n_train, n_test=100, n_mc=10, verbose=True, print_loss=True)


# # Save image and Figure
# # TODO: move to plotting functions, make path arg
# image_path = Path('./images/temp/')
#
# fig = plt.gcf()
# fig.savefig(image_path.joinpath(f"{NOW_STR}"))
# with open(image_path.joinpath(f"{NOW_STR}.mpl"), 'wb') as f:
#     pickle.dump(fig, f)


# model_x = rand.elements.FiniteGeneric([0, .5], p=None)
# model = random.models.DataConditional([random.elements.FiniteGeneric([0, .5], [p, 1 - p]) for p in (.5, .5)], model_x)
# prior_mean = random.models.DataConditional([random.elements.FiniteGeneric([0, .5], [p, 1 - p]) for p in (.9, .9)], model_x)


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
#     prior_mean_x = random.elements.FiniteGeneric(np.linspace(0, 1, n_t), p=_temp / _temp.sum())
#     prior_mean = random.models.BetaLinear(weights=w_prior, basis_y_x=None, alpha_y_x=alpha_y_x_beta,
#                                         model_x=prior_mean_x)
#
#     dir_predictors.append(BayesRegressor(bayes.models.Dirichlet(prior_mean, alpha_0=0.01),
#                                          space=model.space, proc_funcs=[make_discretizer(prior_mean_x.values)],
#                                          name=r'$\mathrm{Dir}$, $|\mathcal{T}| = card$'.replace('card', str(n_t)),
#                                          ))
#
#     if scale_alpha and _params is not None:
#         _params['alpha_0'] *= n_t

# plot_risk_disc(predictors, model_eval, params, n_train, n_test=1, n_mc=50000, verbose=True, ax=None)
# plt.xscale('log', base=2)


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
#         ax.set_xlabel(r'$\alpha_0 / |\mathcal{T}|$ ')
#         _vals = dir_params['alpha_0']
#         ax.set_xlim((min(_vals), max(_vals)))


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

# print(f"\nAnalytical Risk = {opt_predictor.evaluate_analytic(n_train=n_train)}")

# if isinstance(model, random.models.Base):
#     risk_an = opt_predictor.risk_min()
#     print(f"Min risk = {risk_an}")
# elif isinstance(model, bayes.models.Base):
#     risk_an = opt_predictor.bayes_risk_min(n_train)
#     print(f"Min Bayes risk = {risk_an}")
# else:
#     raise TypeError
