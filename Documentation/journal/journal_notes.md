# Outline

## Base structure
- Use both discrete and continuous
  - for math impact and realism
- Bayesian and conditional risk results
  - for math impact
- Discuss bias/variance SE trade-off
  - nice plots, link to math
  - SHOW IN STATS PLOTS!
  - model estimation perspective, too?

## Outstanding goals
- Non-trivial data support for discrete results
  - realism for reader
- Find a sensible application for "countable" continuous data support
  - Needed to make continuous Dirichlet viable
- Compare to classic algorithms (Normal regressor)
  - help reader relate
  - demonstrate Dirichlet consistency vs Normal loss


# Conditional risk results

**REMAKE plots with proper number of values y**

- X_set = Y_set = uniform grid on [0, 1]
- p_x(x) = constant
- p_y_x(x) = BinomialNormalized(n_grid, true_mean(x))

<!-- ## Loss vs N, vs Norm
- 16 point [0, 1] grid
- Normal: 1st order mean, cov_y_x=.1

### biased mean
- true mean: f(x) = x**3
- prior mean: f(x) = .5

![](loss_n_biased_v2.png)

### unbiased mean
- true mean: f(x) = .5
- prior mean: f(x) = .5

![](loss_n_unbiased_v2.png) -->

## Loss vs N, vs Norm, poly
- true mean: f(x) = x**2
- Normal: cov_y_x=.1, prior_cov=10 ?

### unbiased mean
![](predict_unbiased_dir.png)

- Norm order = 2
![](loss_n_unbiased.png)
![](predict_unbiased.png)

### biased mean, f(x) = .5
![](predict_biased_dir.png)

- Norm order = 1
![](loss_n_biased.png)
![](predict_biased.png)
- Norm order = 2
![](loss_n_biased_norm2.png)


## Loss vs N, vs Norm, hard non-poly
- true mean: f(x) = 1 / (2 + sin(2*pi* x))
- Normal: cov_y_x=.1

### biased mean, f(x) = .5
![](predict_biased_hi_dir.png)

- Norm order = 2
![](loss_n_biased_hi.png)
![](predict_biased_hi.png)


## Loss vs alpha_0
- true mean: f(x) = .3 + .4* x**2

### unbiased
![](loss_alpha_unbiased.png)

### biased mean, f(x) = .5
![](loss_alpha_biased.png)


# Bayesian risk results
- set = {0, .1, ... , 1}
- true mean: f(x) = .5
- true alpha_0 = 10

## Loss vs N
- unbiased
![](loss_bayes_n_unbiased.png)

## Loss vs alpha_0
- unbiased
![](loss_bayes_alpha_unbiased.png)
