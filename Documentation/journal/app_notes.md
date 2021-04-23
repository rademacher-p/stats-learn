# CONCEPTS
- Bayesian benefit = different optimization cost function
  - Dirichlet = regularized empirical risk!
  - Parametric learners effect a kind of extreme regularization
- More DOF is fine, if cost is right! Avoid overfitting!

- **DIRICHLET**
  - Least restrictive regularizer (least subjective prior info)
    - Avoid worst case scenarios (via consistency)
    - Mean selection enables preference for any solution (good for low N)
  - Show results across multiple problems (Bayes?) to emphasize robustness?!

Low N: avoid overfitting (*Show L2 reg inferior?*)
High N: consistency (*harder funcs?*)

**MLP universal approx thoerem for CONTINUOUS functions. Discretized Dirichlet seems good for discontinuities! Fractals? RESEARCH!!**

---
# Results
**Do discrete domain results to isolate Dirichlet properties?**

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
**Need harder funcs for pop learners - research!**
**Equate degrees-of-freedom for fair comparison?**

- *Advantages*
  - Full support prior = consistency = best performance for N >> 0
  - Flexible parameterization = able to exploit strong prior knowledge
    - Show pop don't do priors = **Bayes benefit**!!!?
- *Disadvantages*
  - Weak utilization of limited data = outperformed by certain low-dim learners



---
# Real Data notes
Try **SKL** and **UCI** datasets!

- How to select the prior mean?
  - **Boost existing regressors?!?**
- Cross-validation risk should be minimized when using the empirical distribution of the full data set
- Take care to avoid train/test leakage when designing prior mean!
