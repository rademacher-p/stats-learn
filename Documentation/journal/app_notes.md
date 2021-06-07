# Discrete domain CONCEPTS
*Discretization of continuous domain loses consistency, but on discrete domain pop parametric learners can also be consistent?!*
- Pop parametric learners can also be consistent with enough DOF!!
  - Bayesian LR demonstrates
    - Dirichlet has less flexibility? NOT if you consider it as continuous domain case (infinite DOF in mean func)


# CONCEPTS
- Bayesian benefit = different optimization cost function
  - Dirichlet = regularized empirical risk!
  - Parametric learners effect a kind of extreme regularization
- More DOF is fine, if cost is right! Avoid overfitting!

- **DIRICHLET**
  - Least restrictive regularizer (least subjective prior info)
    - Avoid worst case scenarios (via consistency)
    - Mean selection enables preference for any solution (good for low N)
    - *COMPARISON*: only Dir accounts for loss function, pop ignores parameter mapping
  - Show results across multiple problems (Bayes?) to emphasize robustness?!

Low N: avoid overfitting (*Show L2 reg inferior?*)
High N: consistency (*harder funcs?*)

**Need harder funcs for pop learners - research!**
**MLP universal approx theorem for CONTINUOUS functions. Discretized Dirichlet seems good for discontinuities! Fractals? RESEARCH!!**

**Equate degrees-of-freedom for fair comparison? For regression: prior or mean?**


---
# Results

- MODEL
  - Original non-linearity
- Dirichlet(T=16, alpha=10)
  - Original mean; try others if needed!
- MLP(layers=[100, 100, 100, 100], alpha=1e-4, iter=2000, tol=1e-8)
  - *Note regularization*!
  - Investigate # layers/weights
- Best for low N due to MLP overfitting variance


---
# Pop Comparison notes
- *Advantages*
  - Full support prior = consistency = best performance for N >> 0
    - **ERROR: Pop can be consistent on discrete domain with enough DOF!!**
  - Flexible parameterization = able to exploit strong prior knowledge
    - Show pop don't do priors = **Bayes benefit**!!!?
- *Disadvantages*
  - Weak utilization of limited data = outperformed by certain low-dim learners


---
# Real Data notes
Try **SKL** and **UCI** datasets!

- How to select the prior mean?
  - **Boost existing regressors?!?** Not really prior knowledge...
- Note: cross-validation risk should be minimized when using the empirical distribution of the full data set
  - Take care to avoid train/test leakage when designing prior mean!
