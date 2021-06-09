# Discrete domain analysis
- **BayesLR regressor superset of Dirichlet regressor set**
- Dirichlet is a more flexible prior (continuous \alphac), but for regression, the *effective* parameterization is less flexible than that of BayesLR
- Dirichlet, unlike BayesLR, can be applied to classification and other problems


---
- *Hard funcs for pop learners*
  - Random discrete functions
  - Discontinuous funcs vs MLP (UAT only for continuous). Fractals?

**Equate degrees-of-freedom for fair comparison? For regression: prior or mean?**


---
# CONCEPTS
*Preserve consistency while allowing richer regularization (via Bayes, vs L-norm) then most pop learners!?!*

- **DIRICHLET** -> Regularized empirical mean
  - Preserve consistency via full support prior
    - Extreme "regularization" of parametric learners can be detrimental
  - Better regularization via flexible parameterization
    - Enables preference for any solution (good for low N)
    - Avoid overfitting for high variance data models
    - Only Dir accounts for loss function, pop *ignores* parameter mapping

**Fast hyperparameter optimization of Dirichlet localization, no additional training computation!!**


---
# Pop Comparison notes
- *Advantages*
  - Full support prior = consistency = best performance for N >> 0
    - **CRITICAL: BayesLR and NN (ref: Farago)**
  - Flexible parameterization = able to exploit strong prior knowledge
    - **CRITICAL: BayesLR**
- *Disadvantages*
  - Flex = minimal response to limited data = outperformed by certain low-dim learners


---
# Real Data notes
Try **SKL** and **UCI** datasets!

- How to select the prior mean?
  - **Boost existing regressors?!?** Not really prior knowledge...
  - Note: cross-validation risk should be minimized when using the empirical distribution of the full data set
    - Take care to avoid train/test leakage when designing prior mean!


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
