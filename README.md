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