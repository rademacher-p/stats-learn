import numpy as np

from stats_learn import results
from stats_learn.bayes import models as bayes_models
from stats_learn.predictors.base import ModelRegressor, BayesRegressor
from stats_learn.random import models as rand_models

seed = 12345

model = rand_models.NormalLinear(weights=[1, 1])

# Predictors
opt_predictor = ModelRegressor(model, name='Optimal')

norm_model = bayes_models.NormalLinear(prior_mean=[0, 0], prior_cov=1, allow_singular=True)
norm_predictor = BayesRegressor(norm_model, space=model.space, name='Normal')
norm_params = {'prior_cov': [.01, .1]}

# Results
n_test = 1000
n_mc = 100

predictors = [opt_predictor, norm_predictor]
params = [None, norm_params]

# Sample regressor realizations
n_train = 10
d = model.sample(n_train + n_test, rng=seed)
d_train, d_test = np.split(d, [n_train])
results.assess_single_compare(predictors, d_train, d_test, params, verbose=True)

# Prediction mean/variance
n_train = 10
results.assess_compare(predictors, model, params, n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                       plot_stats=True, print_loss=True, rng=seed)

# Squared-Error vs. training data volume
n_train = np.linspace(0, 100, 21, dtype=int)
results.assess_compare(predictors, model, params, n_train, n_test, n_mc, verbose=True, plot_loss=True, print_loss=True,
                       rng=seed)
