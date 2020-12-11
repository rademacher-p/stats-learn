- Use both discrete and continuous
  - for math impact and realism
- Bayesian and conditional risk results
  - for math impact
- Discuss bias/variance SE trade-off
  - nice plots, link to math
  - SHOW IN STATS PLOTS!
  - model estimation perspective, too?

# Primary objectives
- Non-trivial data support for discrete results
  - realism for reader
- Find a sensible application for "countable" continuous data support
  - Needed to make continuous Dirichlet viable
- Compare to classic algorithms (Normal regressor)
  - help reader relate
  - demonstrate Dirichlet consistency vs Normal loss


# Results log

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
- p_x: uniform on 16 point [0, 1] grid
- true mean: f(x) = x**2
- Normal: cov_y_x=.1

### unbiased mean
- Norm order = 2
![](loss_n_unbiased.png)

### biased mean, f(x) = .5
- Norm order = 1
![](loss_n_biased.png)
- Norm order = 2
![](loss_n_biased_norm2.png)


## Loss vs N, vs Norm, hard non-linear
- p_x: uniform on 16 point [0, 1] grid
- true mean: f(x) = 1 / (2 + sin(2*pi* x))
- Normal: cov_y_x=.1

### biased mean, f(x) = .5
- Norm order = 2
![](loss_n_biased_hi.png)


## Loss vs alpha_0
- true mean: f(x) = .5 + .2*x
- prior mean: f(x) = .5

![](loss_alpha_biased.png)
