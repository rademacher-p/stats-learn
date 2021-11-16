# Statistical Learning
This package provides a framework to explore statistical learning with a Bayesian focus.

*Note*: This project is under active development at https://github.com/rademacher-p/stats-learn

## Documentation
Documentation is provided locally at [docs/_build/html/index.html](docs/_build/html/index.html)

## Installation
The package can be installed locally using
```
pip install <path>
```
where the path points to the directory of this README. Note that the
[editable option](https://pip.pypa.io/en/stable/cli/pip_install/) can be used to track any package modifications.

## Examples
Example scripts for reproduction of results are located in the `examples` directory. They can be invoked from the 
command line and given a variety of arguments to control the simulations. A demonstrative help message is shown below, 
along with an exemplifying usage from the package top level.

```
usage: consistency.py [-h] [-m MC] [-l LOG_PATH] [-i] [--style STYLE] [--seed SEED]
                      [{fit,predict_a0,predict_N,risk_N_leg_a0,risk_a0_leg_N} ...]

Example: consistent regressor on a discrete domain

positional arguments:
  {fit,predict_a0,predict_N,risk_N_leg_a0,risk_a0_leg_N}
                        Simulations to run

optional arguments:
  -h, --help            show this help message and exit
  -m MC, --mc MC        Number of Monte Carlo iterations
  -l LOG_PATH, --log_path LOG_PATH
                        Path to log file
  -i, --save_img        Save images to log
  --style STYLE         Path to .mplstyle Matplotlib style
  --seed SEED           RNG seed

```

```commandline
python examples\discrete\regression\consistency.py -m 1000 -l temp/log.md -i 
--style images/style.mplstyle --seed 12345 fit risk_N_leg_a0 
```

Observe that the positional arguments are a variable number of strings specifying which images to generate. The `--mc` 
option allows control over how many simulated datasets are generated for statistically meaningful results; the `--seed` 
option controls random number generation for reproducibility. 

The `--log_path` and `--save_img` options enable logging 
of result tables and/or images into a Markdown-format file for future use; note that the specified log path file will 
be created if it does not exist. Additionally, note that a specific [Matplotlib](https://matplotlib.org/) `--style` can 
be specified for custom formatting. 

To implement the same formatting used throughout the publication, a provided style
can be used (as shown above); note that this style requires [LaTeX](https://www.latex-project.org/), as well as the 
[bm](https://www.ctan.org/pkg/bm) and [upgreek](https://www.ctan.org/pkg/upgreek) packages.

## Example

```python
import numpy as np
from matplotlib import pyplot as plt

from stats_learn import random, bayes, results
from stats_learn.predictors.base import ModelRegressor, BayesRegressor

seed = 12345
plt.style.use('images/style.mplstyle')

model = random.models.NormalLinear(weights=[1, 1])

# Predictors
opt_predictor = ModelRegressor(model, name='Optimal')

norm_model = bayes.models.NormalLinear(prior_mean=[0, 0], prior_cov=1, allow_singular=True)
norm_predictor = BayesRegressor(norm_model, name='Normal')
norm_params = {'prior_cov': [.01, .1]}

# Results
n_test = 10
n_mc = 10

predictors = [opt_predictor, norm_predictor]
params = [None, norm_params]

# Sample regressor realizations
n_train = 10
d = model.sample(n_train + n_test, rng=seed)
d_train, d_test = np.split(d, [n_train])
results.data_assess(predictors, d_train, d_test, params, verbose=True)

# Prediction mean/variance
n_train = 10
results.model_assess(predictors, model, params, n_train, n_test, n_mc, stats=('mean', 'std'), verbose=True,
                     plot_stats=True, print_loss=True, rng=seed)

# Squared-Error vs. training data volume
n_train = np.linspace(0, 100, 21, dtype=int)
results.model_assess(predictors, model, params, n_train, n_test, n_mc, verbose=True, plot_loss=True, rng=seed)
```