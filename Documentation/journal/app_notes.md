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

**Combine Dirichlet with Discretization transforms for more subjective prior knowledge?**


---
# Pop Comparison notes
**ADD more advanced estimators to my Example for legitimacy**
**Large N comparison table to show where Dirichlet is best?**
**Need harder funcs for pop learners - research!**
**Equate degrees-of-freedom for fair comparison?**
*Investigate where MLP fails due to overfitting (low N?) MODEL VARIANCE?*

- *Advantages*
  - Full support prior = consistency = best performance for N >> 0
  - Flexible parameterization = able to exploit strong prior knowledge
    - Show pop don't do priors = **Bayes benefit**!!!?
- *Disadvantages*
  - Weak utilization of limited data = outperformed by certain low-dim learners


[0,1]^2^N
(T x [0,1])^N
R^T
R^p


---
# Real Data notes
Try **SKL** and **UCI** datasets!

- How to select the prior mean?
  - **Boost existing regressors?!?**
- Cross-validation risk should be minimized when using the empirical distribution of the full data set
- Take care to avoid train/test leakage when designing prior mean!
