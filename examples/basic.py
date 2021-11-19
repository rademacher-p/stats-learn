# from matplotlib import pyplot as plt

from stats_learn import random, bayes, results
from stats_learn.predictors.base import ModelRegressor, BayesRegressor

# plt.style.use('images/style.mplstyle')

model = random.models.NormalLinear(weights=[1, 1])

# Predictors
opt_predictor = ModelRegressor(model, name='Optimal')

norm_model = bayes.models.NormalLinear(prior_mean=[0, 0], prior_cov=1)
norm_predictor = BayesRegressor(norm_model, name='Normal')

# Results
seed = 12345
n_train = 10
n_test = 20

d = model.sample(n_train + n_test, rng=seed)
d_train, d_test = d[:n_train], d[n_train:]

loss_min = opt_predictor.evaluate(d_test)
print(f"Minimum loss = {loss_min:.3f}")

loss_prior = norm_predictor.evaluate(d_test)  # use the prior distribution
print(f"Untrained learner loss = {loss_prior:.3f}")

norm_predictor.fit(d_train)  # fit the posterior distribution
loss_fit = norm_predictor.evaluate(d_test)
print(f"Trained learner loss = {loss_fit:.3f}")


# Results
predictors = [opt_predictor, norm_predictor]
params = [None, {'prior_cov': [.01, .1, 1]}]

# Sample regressor realizations
results.data_assess(predictors, d_train, d_test, params, verbose=True, plot_fit=True)

# Prediction mean/variance
results.model_assess(predictors, model, params, n_train, n_test, n_mc=1000, stats=('mean', 'std'), verbose=True,
                     plot_stats=True, print_loss=True, rng=seed)

# Squared-Error vs. training data volume
n_train = range(0, 100, 5)
results.model_assess(predictors, model, params, n_train, n_test, n_mc=1000, verbose=True, plot_loss=True, rng=seed)
