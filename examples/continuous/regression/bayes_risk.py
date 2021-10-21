import numpy as np
from matplotlib import pyplot as plt

from stats_learn.util.base import get_now
from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors.base import BayesRegressor

plt.style.use('../../../images/style.mplstyle')

# seed = None
seed = 12345

# log_path = None
# img_path = None

# TODO: remove path stuff and image names below before release
base_path = __file__[__file__.rfind('/')+1:].removesuffix('.py') + '_temp/'
log_path = base_path + 'log.md'
img_dir = base_path + f'images/{get_now()}/'


# %% Model and optimal predictor
var_y_x_const = 1/5
w_model = [.5]

alpha_y_x = 1/var_y_x_const - 1
model_x = rand_elements.Uniform([0, 1])
prior_mean = rand_models.BetaLinear(weights=w_model, alpha_y_x=alpha_y_x, model_x=model_x)

model = bayes_models.Dirichlet(prior_mean, alpha_0=4e2)


# %% Dirichlet Learner
dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)
dir_predictor = BayesRegressor(dir_model, name=r'$\mathrm{Dir}$')


# %% Results
n_test = 100
n_mc = 1000

# Bayes Squared-Error vs. N
n_train = np.arange(0, 4050, 50)
dir_params = {'alpha_0': [40, 400, 4000]}

img_path = img_dir + 'risk_bayes_N_leg_a0.png'
y_stats_full, loss_full = dir_predictor.assess(model, dir_params, n_train, n_test, n_mc,
                                               verbose=True, plot_loss=True, print_loss=False,
                                               log_path=log_path, img_path=img_path, rng=seed)

