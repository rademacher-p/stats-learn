import math

import numpy as np
from matplotlib import pyplot as plt

from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors import ModelRegressor, BayesRegressor
from stats_learn.util import funcs, results
from stats_learn.util.data_processing import make_clipper
from stats_learn.util.plotting import box_grid

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{bm,upgreek}")

# seed = None
seed = 12345


#%% Model and optimal predictor
n_x = 128

shape_x = ()
size_x = math.prod(shape_x)
lims_x = np.broadcast_to([0, 1], (*shape_x, 2))
supp_x = box_grid(lims_x, n_x, endpoint=True)
model_x = rand_elements.Finite(supp_x, p=np.full(size_x*(n_x,), n_x**-size_x))

nonlinear_model = funcs.make_inv_trig(shape_x)
var_y_x_const = 1/5

alpha_y_x = (1-var_y_x_const) / (np.float64(var_y_x_const) - 1/(n_x-1))
model = rand_models.DataConditional.from_func_mean(n_x, alpha_y_x, nonlinear_model, model_x, rng=seed)

opt_predictor = ModelRegressor(model, name=r'$f_{\Theta}(\theta)$')


# do_bayes = False
# # do_bayes = True
# if do_bayes:
#     model_eval = bayes_models.Dirichlet(deepcopy(model), alpha_0=4e2)
#     opt_predictor = BayesRegressor(model_eval, name=r'$f^*$')
# else:
#     model_eval = model
#     opt_predictor = ModelRegressor(model_eval, name=r'$f_{\Theta}(\theta)$')
#
# model_eval.rng = seed


#%% Bayesian learners
w_prior = [.5, 0]

# Dirichlet learner
prior_mean = rand_models.DataConditional.from_poly_mean(n_x, alpha_y_x, w_prior, model_x)
dir_model = bayes_models.Dirichlet(prior_mean, alpha_0=10)

dir_predictor = BayesRegressor(dir_model, space=model.space, name=r'$\mathrm{Dir}$')

dir_params = {'alpha_0': [10, 1000]}

# if do_bayes:  # add true bayes model concentration
#     if model_eval.alpha_0 not in dir_params['alpha_0']:
#         dir_params['alpha_0'] = np.sort(np.concatenate((dir_params['alpha_0'], [model_eval.alpha_0])))


# Normal learner
proc_funcs = {'pre': [], 'post': [make_clipper(lims_x)]}

norm_model = bayes_models.NormalLinear(prior_mean=w_prior, prior_cov=.1, cov_y_x=.1, model_x=model_x,
                                       allow_singular=True)
norm_predictor = BayesRegressor(norm_model, space=model.space, proc_funcs=proc_funcs, name=r'$\mathcal{N}$')

norm_params = {'prior_cov': [.1, .001]}


#%% Results
n_train = 400
n_test = 1000
n_mc = 5


temp = [
    (opt_predictor, None),
    (dir_predictor, dir_params),
    (norm_predictor, norm_params),
]
predictors, params = zip(*temp)


y_stats_full, loss_full = results.predictor_compare(predictors, model, params, n_train, n_test, n_mc,
                                                    stats=('mean', 'std'), plot_stats=True, print_loss=True,
                                                    verbose=True)

# y_stats_full, loss_full = results.predictor_compare(predictors, model, params, n_train, n_test, n_mc,
#                                                     plot_loss=True, print_loss=True,
#                                                     verbose=True)