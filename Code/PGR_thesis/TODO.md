# TODO's

- DOCUMENTATION
- analytical risk?
- Computational burden?


# External learner integration

- Learning
  - [x] `fit`, `predict`, `evaluate`
  - [ ] **Need support for learners that cannot warm start?**
    - Logic in MC funcs to pass all previous data
  - [ ] `set_params`
  
- Plotting
  - [ ] Avoid `tex_params`
    - Make a registry for `str` mappings?
  - [x] Preserve `name` attribute?
  - [x] Attr `space_x` used for predict plots
    - **Make it a func argument instead, deprecate `x`?!?**
    - [x] **Remove predictor `space` arg?**