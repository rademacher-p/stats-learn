import numpy as np
from matplotlib import pyplot as plt
import torch
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything

from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors.base import ModelRegressor, BayesRegressor
from stats_learn.util import funcs
from stats_learn import results
from stats_learn.util.data_processing import make_clipper
from stats_learn.predictors.torch import LitMLP, LitWrapper, reset_weights

plt.style.use('../../../style.mplstyle')

# seed = None
seed = 12345

# file = None
# img_path = None
file = '../../../logs/temp/temp.md'
img_path = '../../../images/temp/hi.png'

if seed is not None:
    seed_everything(seed)  # PyTorch-Lightning seeding


#%% Model and optimal predictor
n_x = n_y = 128
nonlinear_model = funcs.make_rand_discrete(n_x, rng=seed)
var_y_x_const = 1/125

supp_x = np.linspace(0, 1, n_x, endpoint=True)
model_x = rand_elements.Finite(supp_x, p=None)

alpha_y_x = (1-var_y_x_const) / (np.float64(var_y_x_const) - 1/(n_y-1))
model = rand_models.DataConditional.from_func_mean(n_y, alpha_y_x, nonlinear_model, model_x)

opt_predictor = ModelRegressor(model, name=r'$f_{\Theta}(\theta)$')


#%% Learners

# Dirichlet
prior_mean = rand_models.DataConditional.from_func_mean(n_y, alpha_y_x, lambda x: .5, model_x)
dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)

dir_predictor = BayesRegressor(dir_model, space=model.space, name=r'$\mathrm{Dir}$')
dir_params = {'alpha_0': [1e-5, 1e5]}


# PyTorch
weight_decays = [0., 1e-3]  # controls L2 regularization

proc_funcs = {'pre': [], 'post': [make_clipper([np.min(supp_x), np.max(supp_x)])]}

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
        'logger': pl_loggers.TensorBoardLogger('../../../logs/learn/', name=logger_name),
        'weights_summary': None,
        'gpus': torch.cuda.device_count(),
    }

    lit_model = LitMLP([model_x.space.size, *layer_sizes], optim_params=optim_params)

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

# Sample regressor realizations
n_train = 20

d = model.rvs(n_train + n_test, rng=seed)
d_train, d_test = np.split(d, [n_train])
loss_full = results.plot_fit_compare(predictors, d_train, d_test, params, img_path=img_path, file=file)

# Prediction mean/variance, comparative
n_train = 128
y_stats_full, loss_full = results.assess_compare(predictors, model, params, n_train, n_test, n_mc,
                                                 stats=('mean', 'std'), verbose=True, plot_stats=True, print_loss=True,
                                                 img_path=img_path, file=file, rng=seed)

# Squared-Error vs. training data volume N
n_train = np.insert(2**np.arange(11), 0, 0)
y_stats_full, loss_full = results.assess_compare(predictors, model, params, n_train, n_test, n_mc, verbose=True,
                                                 plot_loss=True, print_loss=True, img_path=img_path, file=file, rng=seed)
