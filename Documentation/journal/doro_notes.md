# 4.1.2
- [x] eq 4.1, eq 4.3: lose geometric?
  - A: Made a note. Which form IS used? Does it hurt?
- [x] eq 4.6: divide by zero! *specify domain*
  - A: Added domain set dependent on data to avoid undefined case
  - *Union of Xn*!?
  - Use PMF, finite subset of Ycal?

- [x] semicolon confusing? Use bold/case to distinguish?
  - A: Keeping as is, or switch to serial brackets? Bold/case common for linear algebra, but not for analysis - may be confusing.
- [x] dirac vs kron, use diff. font!? partial delta?
  - A: Left for now. Maybe set subscripts for Dirac? Index subscripts for Kronecker, or maybe use Iverson bracket instead?


# 4.2
- [ ] **divide by dirac zero**!?
  - Make PDF/delta(0) into CDF(x)-CDF(x-)??
  - *Exclusion of conditional continuous RP* forms won't eliminate delta(0)!
    - Appearance of delta(0) terms seem to happen naturally (see eq 4.30) even without new DEP work
    - Same in cond. risk via PDF estimate variance (see my notes)
  - CDF difference mod *wouldn't synergize with discrete* (general non-numeric) forms...
  - **Use of Kronecker to avoid delta(0) requires Dirac-to-Kron assumption in eq. 4.6!!**
    - If divide by delta(0) is wrong, so is that equivalence...
  - **Consider going back to counts for marginal, avoids lots of delta(0)? Also removes much of the discrete/continuous form difference??**
    - Fails to address continuous conditionals for Dirichlet??
  - My approach use (https://mathoverflow.net/questions/273965/was-there-ever-proposed-a-theory-where-the-value-of-dirac-delta-at-zero-had-mean)


- [x] dirac amplitude doesn't necessarily tend to infinity, think sinc func?
  - A: Is this accurate? Infinitesimal support seems to necessitate infinite amplitude. Sinc func example must be scaled up!?


# 4.3
- [x] plot PDFs?
  - A: Max bias, zero var makes for uninformative plots

# 5
- [x] "severe drawbacks"
  - A: Changed
- [x] "cardinality reducing"
  - A: Changed.
- [x] contiguous vs continuous vs connected
  - A: Switched to "connected spaces"!
- [x] discretization vs quantization
  - A: Former is more common in ML terminology, code, etc.
- [x] use projection terms
  - A: Inappropriate, transform is non-linear?!
- [x] PDF tends in what sense?
  - A: Different types are for RV convergence, not function convergence?
