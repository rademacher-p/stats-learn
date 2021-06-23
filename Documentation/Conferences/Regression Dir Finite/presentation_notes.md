- [x] Describe sim example
  - *Use thesis eqs and figs!*
- [x] Predict stats fig placement?
  - After formulae for context? Or up front to illustrate example?
- [x] Move P_y_x equivalence earlier?
- [x] Check for missing citations
  - Multiple footnote cite error?
  - *OK with excluded ones*
- [x] Check for residual Drm

---
**Stop saying "of course" and "really"**

---
# Talking Points
- The Dirichlet prior
  - *Universal* consistency
  - Flexible and *interpretable* parameterization
- Sufficient Statistics
  - *Likelihood function* shows dependency through `\psi`
  - Ensures we can use `\psi` without loss of information
- Predictive Model Posterior - Closed-form
  - Localization control affects weights
- Predictive Model Posterior - Trends
  - Not only does the mean tend, the entire distribution concentrates!
- Bayes Optimal Regressor
  - Regressor inherits trends from posterior dist.
- Prediction Statistics
  - Normal regressor is always biased due to lower-dim, limited-support prior
  - "A trade-off is shown, which is formalized next"
- Optimal Localization
  - Trends with bias and model variance (OVERFITTING)
  - Optimal value independent of N
  - May not be able to use formula in practice, but next slide shows consistency
- Training Volume Trends
  - "Universal consistency"
