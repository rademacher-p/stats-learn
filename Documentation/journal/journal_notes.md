# Outline

## Base structure
- Use both discrete and continuous
  - for math impact and realism
- Bayesian and conditional risk results
  - for math impact
- Discuss bias/variance SE trade-off
  - nice plots, link to math
  - model estimation perspective, too?
- **Real data APPLICATION**
  - Discretization of continuous data!?


## Outstanding goals
- Non-trivial data support for discrete results
  - realism for reader
- Find a sensible application for "countable" continuous data support
  - Needed to make continuous Dirichlet viable
    - **Discretization**
- Compare to classic algorithms (Normal regressor)
  - help reader relate
  - demonstrate Dirichlet consistency vs Normal loss


# Discrete set results

- X_set = Y_set = uniform grid on [0, 1]
- p_x(x) = constant
- p_y_x(x) = EmpiricalScalar(n_grid, true_mean(x))


## Conditional risk

**REMAKE plots with proper number of values y**

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

### Loss vs N, vs Norm, poly
- true_mean(x) = x**2
- Normal: cov_y_x=.1, prior_cov=10 ?

#### unbiased mean
![](predict_unbiased_dir.png)

- Norm order = 2
![](loss_n_unbiased.png)
![](predict_unbiased.png)

#### biased mean, f(x) = .5
![](predict_biased_dir.png)
![](predict_biased_dir_a0.png)

- Norm order = 1
![](loss_n_biased.png)
![](predict_biased.png)
- Norm order = 2
![](loss_n_biased_norm2.png)


### Loss vs N, vs Norm, hard non-poly
- true_mean(x) = 1 / (2 + sin(2*pi* x))
- Normal: cov_y_x=.1

#### biased mean, f(x) = .5
![](predict_biased_hi_dir.png)
![](predict_biased_hi_dir_a0.png)


- Norm order = 2
![](loss_n_biased_hi.png)
![](predict_biased_hi.png)


### Loss vs alpha_0
- true_mean(x) = .3 + .4* x**2

#### unbiased
![](loss_alpha_unbiased.png)

#### biased mean, f(x) = .5
![](loss_alpha_biased.png)


## Bayesian risk
- true_mean(x) = .5
- true alpha_0 = 10
- NOTE: set = {0, .1, ... , 1}


### Loss vs N
- unbiased
![](loss_bayes_n_unbiased.png)

### Loss vs alpha_0
- unbiased (note correct minima)
![](loss_bayes_alpha_unbiased.png)



# Continuous set results

- X_set = Y_set = uniform on [0, 1]
- p_x(x) = 1 (constant)
- p_y_x(x) = Beta(alpha_y_x * true_mean(x), alpha_y_x * (1 - true_mean(x)))
  - alpha_y_x = 100

## Conditional risk
### Loss vs N, vs Norm, poly
- true_mean(x) = x**2
- Normal: cov_y_x=.1, prior_cov=10 ?

#### biased mean, f(x) = .5
NOTE: plotting missed at training data points
![](cont_predict_biased_dir.png)

- Norm order = 1
![](cont_loss_n_biased.png)


## Bayesian risk
- true_mean(x) = .5
- true alpha_0 = 10


### Loss vs N
- unbiased
![](cont_loss_bayes_n_unbiased.png)

### Loss vs alpha_0
- unbiased
![](cont_loss_bayes_alpha_unbiased.png)
