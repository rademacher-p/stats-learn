- **Show overfitting example vs NN with differing degrees of regularization**
  - NN w/o reg. = overfitting
  - NN w/ reg. = insufficiently flexible/intuitive?
  - Best case: Dirichlet empirical is worst (*high N*) via MAX overfitting, then empirical NN (inherent reg.), then L2 reg. NN, then informative Dirichlet for MAX reg.

- **Show underfitting example vs NN due to inherent regularization?!**
  - NN w/o reg. = mild underfitting
  - NN w/ reg. = severe underfitting
  - **Use lower variance model??**

- Clairvoyant regressors
  - Random func is *hard for NN*, but makes it difficult to use meaningful prior mean
    - High variation makes regularization *worse* for both Dir and NN!!
      - Dirichlet **benefit** = purest empirical learner for high variance data?!
  - Discontinuous funcs? (NN UAT only holds for continuous)


- Need better NN architectures and optimizers to *maximize* overfitting!!
  - Comment on **excessive** DOF and train time for DNN!


- If |X| >> N, the prior mean will tend to be used with high probability and the effects of alpha_0 will be negligible. Bad for demonstrating empirical Dirichlet.
  - Should use *higher* N to contrast localization effects!?

- Use smaller discrete spaces? Or use continuous space and **discretization**?
  - Discretization makes the *restricted* function space a major regularizer!
  - **DISCRETE** enables better consistency/overfitting trade demo!!!



---
- Equate degrees-of-freedom for fair comparison? For regression: prior or mean?
  - Related to computation -> not really in scope of my analysis

*Fast hyperparameter optimization of Dirichlet localization, no additional training computation!!*

---
# Discrete domain analysis
- **BayesLR regressors `\superset` Dirichlet regressors**
  - LR, Ridge are subsets of BayesLR, but still used -> motivates use of Dirichlet
  - Dirichlet is a more flexible prior (continuous `\alphac`), but for regression, the *effective* parameterization is less flexible than that of BayesLR
  - Dirichlet, unlike BayesLR, can be applied to classification and other problems


---
# CONCEPTS
*Preserve consistency while allowing richer regularization (via Bayes, vs L-norm) then most pop learners!?!*

- **DIRICHLET** -> Regularized empirical mean
  - Preserve consistency via full support prior
    - Extreme "regularization" of parametric learners can be detrimental
  - Better regularization via flexible parameterization
    - Avoid overfitting for high variance data models
    - Enables preference for *any* solution (good for low N)
    - Only Dir accounts for loss function, pop *ignores* parameter mapping


---
# Pop Comparison notes
- *Advantages*
  - Full support prior = consistency = opt. performance for N >> 0
    - **CRITICAL: BayesLR and NN (ref: Farago)**
  - Flexible parameterization = able to exploit strong prior knowledge
    - **CRITICAL: BayesLR**
- *Disadvantages*
  - Flex = minimal response to limited data = outperformed by certain low-dim learners


---
# Real Data notes
Try **SKL** and **UCI** datasets!

- How to select the prior mean?
  - **Boost existing regressors?!?** = not really prior knowledge...
  - Note: cross-validation risk should be minimized when using the empirical distribution of the full data set
    - Take care to avoid train/test leakage when designing prior mean!
