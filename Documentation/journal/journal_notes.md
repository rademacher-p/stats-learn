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


## Real Data notes
- How to select the prior mean?
  - **Boost existing regressors?!?**
- Cross-validation risk should be minimized when using the empirical distribution of the full data set
- Take care to avoid validation bias when designing prior mean!
