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
Example scripts for reproduction of results are located in the `examples` directory.

To replicate the figures using LaTeX formatting... need latex, packages...

log path doesn't have to exist

```
python examples\discrete\regression\argparse_consistency.py
-l=temp/log.md -i --style=images/style.mplstyle --seed=12345
```