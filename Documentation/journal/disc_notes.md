# Tasks
- predict stats
  - [x] fixed N, various alpha (add gaussian)
  - [x] fixed alpha, various N
  - [x] *fixed N, fixed alpha, various T*
    - discretization bias
- loss
  - [x] vs N, various alpha (add gaussian)
  - [x] *vs N, various T (add gaussian)*
    - different asymptotes, different convergence speed
  - [x] vs alpha, various N
  - [] *vs alpha, various T*
    - different argmins **add min markers to plot?!**
  - [] *vs T, various N*
    - demonstrate trade-off


# Notes
- Include an a0 legend fig to show gaussian?
- SE_predict_leg_T
  - Coarse has low variance, no bias relative to thetacd, but high disc bias. Fine has high bias relative to thetacd because of insufficient data. Middle optimizes the trade-off.
  - *move alpha from legend to title*

- SE_predict_leg_N_t4_v2
  - Convergence to thetacd (no related bias), no variance, but disc bias remains. Contrast with discrete-domain consistency plot.

- SE_risk_N_leg_T
  - Coarse has fastest convergence, but worst asymptote. Fine can barely adapt, so perfect asymptote is never realized. Middle optimizes the trade-off for middle N values.
  - Gaussian demonstrates same trade-off!
  - *add more gaussian or just remove*
  - *move alpha from legend to title*
