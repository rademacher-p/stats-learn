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


---
# CONCEPTS
- Bayesian benefit = different optimization cost function
  - Dirichlet = regularized empirical risk!
  - Parametric learners effect a kind of extreme regularization
- More DOF is fine, if cost is right! Avoid overfitting!
- Quality of regularizer is model dependent

- **DIRICHLET**
  - Least restrictive regularizer (least subjective prior info)
    - Avoid worst case scenarios (esp. for large N)
  - alpha_0 is regularizing term weight
    - Combines to enable strong preference for any solution (good for small N)
  - **Show results across multiple problems to emphasize robustness benefit!**
    - Bayes risk??
    - Use large N for consistency effect
    - Best flexibility when problem is not understood


---
# Pop Comparison notes
**ADD more advanced estimators to my Example for legitimacy**
**Large N comparison table to show where Dirichlet is best**

- *Advantages*
  - Full support prior = consistency = best performance for N >> 0
  - Flexible parameterization = able to exploit strong prior knowledge
    - Show pop don't do priors = **Bayes benefit**!!!?
- *Disadvantages*
  - Weak utilization of limited data = outperformed by certain low-dim learners

**Equate degrees-of-freedom for fair comparison?**
**Need harder funcs for pop learners - research!**

[0,1]^2^N
(T x [0,1])^N
R^p


---
# Real Data notes
Try **SKL** and **UCI** datasets!

- How to select the prior mean?
  - **Boost existing regressors?!?**
- Cross-validation risk should be minimized when using the empirical distribution of the full data set
- Take care to avoid validation bias when designing prior mean!
