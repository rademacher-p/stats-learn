# Tasks
- predict stats
  - [x] fixed N, various alpha (add gaussian)
  - [x] fixed alpha, various N
  - [x] *fixed N, fixed alpha, various T*
- loss
  - [x] vs N, various alpha (add gaussian)
  - [x] *vs N, various T (add gaussian)*
  - [x] vs alpha, various N
  - [x] *vs alpha, various T*
  - [x] *vs T, various N*
    - demonstrate trade-off and argmin dependency on N
  - [x] *vs T, various alpha*
    - sims suggest same argmin??


# Notes
- SE_predict_leg_T
  - Coarse has low variance, low bias relative to thetacd, but high disc bias. Fine has high bias relative to thetacd because of insufficient data, but low disc bias. Middle optimizes the trade-off.

- SE_predict_leg_N_t4_v2
  - Convergence to thetacd (no related bias), no variance, but disc bias remains. Contrast with discrete-domain consistency plot.

- SE_risk_N_leg_T
  - Coarse has fastest convergence (data sensitivity), but worst asymptote (disc bias). Fine can barely adapt, so perfect asymptote is never realized in practice. Middle optimizes the trade-off for middle N values.
  - Gaussian demonstrates same trade-off!

- SE_risk_a0norm_leg_T
  - different argmins
  - sims show lower T -> lower opt alpha (due to summation effect...)
    - same R_min, same a0bar, lower summed a0
    - **higher alpha/|T|** = average conditional concentration!!
