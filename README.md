# Statistical Learning
![CI](https://github.com/rademacher-p/stats-learn/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/github/rademacher-p/stats-learn/branch/main/graph/badge.svg?token=4U2OKN862G)](https://codecov.io/github/rademacher-p/stats-learn)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
<!-- [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) -->
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6886844.svg)](https://doi.org/10.5281/zenodo.6886844)

> **Note**: This project is under active development. :construction:

This Python package provides a framework to explore statistical learning with a Bayesian focus. It implements a variety of `random.elements`, as well as `random.models` of data for supervised learning. The `bayes` subpackage implements similar elements/models with `prior` attributes to statistically characterize parameter uncertainty and `fit` methods to adapt posteriors.

For supervised learning, the `predictors` subpackage provides objects that use these statistical models to define inference and decision functions. Additionally, customization enables comparison with learning objects from popular machine learning packages. The `predictors.torch` submodule uses [PyTorch](https://pytorch.org/) (and [PyTorch Lightning](https://www.pytorchlightning.ai/)) to implement neural networks in the `stats_learn` API.

Also included (in the `results` submodule) are various functions that enable fair and reproducible evaluations, as well as provide visualizations and Markdown-formatted output. Furthermore, they allow efficient assessments for learners across a set of hyperparameter values.

## Installation
The `stats_learn` package is developed for [Python](https://www.python.org/downloads/) 3.8-3.10. Best practice is to first create a [virtual environment](https://docs.python.org/3/tutorial/venv.html). The package can then be installed directly from GitHub using
```
pip install git+https://github.com/rademacher-p/stats-learn
```
To install a specific version or branch, [specify a ref](https://pip.pypa.io/en/stable/topics/vcs-support/). Alternatively, the package can be installed locally using `pip install <path>`, where `<path>` is the top-level directory containing `pyproject.toml`.
Note that with both methods, the [editable option](https://pip.pypa.io/en/stable/cli/pip_install/) can be included to track any package modifications. To install optional packages, specify [extras](https://peps.python.org/pep-0508/#extras), as exemplified [here](https://pip.pypa.io/en/stable/cli/pip_install/#examples). Developers should install the package with `pip install -e <path>[dev]`. The additional commands below should be run on new environments, activating formatting/linting [hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) and [static typing](https://mypy.readthedocs.io/en/stable/index.html) support:
```
pip install --upgrade pip
pre-commit install
mypy --install-types --non-interactive .
```

The project management tool [hatch](https://hatch.pypa.io/) is recommended to simplify local installation. Simply execute `hatch shell` to create and enter a virtual environment with the package installed in development mode (editable install plus extras). Post-install commands (for `pre-commit`, `mypy`, etc.) are executed automatically.


## Documentation
Package documentation is available [here](https://rademacher-p.github.io/stats-learn/).

Alternatively, the docs can be generated using the `sphinx` package and the `sphinx-rtd-theme`, both installable using `pip`. To build the HTML documentation, run `make html` from the `docs/` folder; the top level document will be `docs/build/html/index.html`

## Quickstart
A basic example of model creation, learner definition, and performance assessment is shown below. The `model` attribute defines a jointly Normal distribution where the expected value of `y` conditioned on `x` is characterized by a polynomial function.

Two different predictors are instantiated. First, the `opt_predictor` uses knowledge of the `model` to determine the optimal `predict` function. Second, a learning regressor is formulated using a Bayesian data model `norm_model`; this object implements a Normal distribution `norm_model.prior` to characterize uncertainty about the true model `weights`.

Training and testing data are randomly generated using the model `sample` method and each predictor is assessed using its `evaluate` method. Once the learning `norm_predictor` is `fit` to the data, its squared-error loss is reduced.

```python
from stats_learn import bayes, random
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
```

Output:
```
Minimum loss = 0.549
Prior learner loss = 3.413
Trained learner loss = 0.951
```

The code below can be executed after the previous snippet. The `data_assess` function provides replication of the functionality above, including a loss table and a graphic showing how the `predict` functions fit the training data. The `model_assess` function performs Monte Carlo approximation of the expected loss by repeatedly generating and evaluating on new datasets, enabling statistically meaningful evaluation. Observe that it can be used for both visualization of both the prediction statistics and of the average loss.

```python
from stats_learn import results

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
```

Output:

|                                 | 10    |
| ------------------------------- | ----- |
| Optimal                         | 0.549 |
| Normal, $\Sigma_\theta = 0.010$ | 3.171 |
| Normal, $\Sigma_\theta = 0.100$ | 2.034 |
| Normal, $\Sigma_\theta = 1.000$ | 0.951 |

![Fitting](images/ex_fit.png)

|                                 | 10    |
| ------------------------------- | ----- |
| Optimal                         | 1.005 |
| Normal, $\Sigma_\theta = 0.010$ | 2.689 |
| Normal, $\Sigma_\theta = 0.100$ | 1.629 |
| Normal, $\Sigma_\theta = 1.000$ | 1.205 |

![Prediction statistics](images/ex_stats.png)

![Empirical risk](images/ex_loss.png)
