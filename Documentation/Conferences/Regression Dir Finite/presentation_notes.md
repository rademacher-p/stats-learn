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
**Pronounce "Dee-ree-shlay"**

---
# Talking Points
- The Dirichlet prior
  - *Universal* consistency; lower-dim parametric learners cannot guarantee!
  - Closed-form = computationally tractable
  - Flexible and *interpretable* parameterization
- Data Representation
  - Note that sets are discrete
  - Parameters are not limited, but the entire distribution!
- Sufficient Statistics
  - *Likelihood function* shows dependency through `\psi`
  - Ensures we can use `\psi` without loss of information
- Predictive Model Posterior - Closed-form
  - Localization control affects weights
- Predictive Model Posterior - Trends
  - Not only does the mean tend, the entire distribution concentrates!
- Bayes Optimal Regressor
  - Predictive dist is mix of dists *->* estimate is mix of first moments
- Example
  - Simulated in my novel Python package
- Prediction Statistics
  - Normal regressor is always biased due to lower-dim, limited-support prior
  - "A trade-off is shown, which is formalized next"
- Optimal Localization
  - Trends with bias and model variance (OVERFITTING)
  - Optimal value independent of N
  - May not be able to use formula in practice, but next slide shows consistency
- Training Volume Trends
  - "Universal consistency"
