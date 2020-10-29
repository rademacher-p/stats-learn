# TODO's

- ANALYTICAL RISK FUNCS
- Computational burden?
- Empirical dist func objects?

## Continuous dists
- Validate with Beta example. Done?
- Deterministic prior with DEP to effect DP realization and sample

Rework YcX stuff for functions?


 ## Dirichlet Bayes
- `mean` parameter = RE object
- `_rvs` = DEP sampling from mean RE
- `random_element` = N/A for continuous
  - Implement for Finite in subclass?
- `posterior` = N/A for continuous?
- `posterior_model` = mixture dist?
  - Create `mixture` RE class
