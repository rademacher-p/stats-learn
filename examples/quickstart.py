from stats_learn import bayes, random, results
from stats_learn.loss_funcs import loss_se
from stats_learn.predictors import BayesRegressor, ModelRegressor

loss_func = loss_se
model = random.models.NormalLinear(weights=[1, 1])

# Predictors
opt_predictor = ModelRegressor(model, name="Optimal")

norm_model = bayes.models.NormalLinear(prior_mean=[0, 0], prior_cov=1)
norm_predictor = BayesRegressor(norm_model, name="Normal")

# Results
seed = 12345
n_train = 10
n_test = 20

d = model.sample(n_train + n_test, rng=seed)
d_train, d_test = d[:n_train], d[n_train:]

loss_min = results.evalutate(opt_predictor, loss_func, d_test)
print(f"Minimum loss = {loss_min:.3f}")

loss_prior = results.evaluate(norm_predictor, loss_func, d_test)
print(f"Untrained learner loss = {loss_prior:.3f}")

norm_predictor.fit(d_train)
loss_fit = results.evaluate(norm_predictor, loss_func, d_test)
print(f"Trained learner loss = {loss_fit:.3f}")


predictors = [opt_predictor, norm_predictor]
params = [None, {"prior_cov": [0.01, 0.1, 1]}]

# Sample regressor realizations
results.data_assess(
    predictors,
    loss_func,
    d_train,
    d_test,
    params,
    verbose=True,
    plot_fit=True,
    img_path="fit.png",
)

# Prediction mean/variance
results.model_assess(
    predictors,
    loss_func,
    model,
    params,
    n_train,
    n_test,
    n_mc=1000,
    stats=("mean", "std"),
    verbose=True,
    plot_stats=True,
    print_loss=True,
    img_path="stats.png",
    rng=seed,
)

# Squared-Error vs. training data volume
n_train_vec = range(0, 100, 5)
results.model_assess(
    predictors,
    loss_func,
    model,
    params,
    n_train_vec,
    n_test,
    n_mc=1000,
    verbose=True,
    plot_loss=True,
    img_path="loss.png",
    rng=seed,
)
