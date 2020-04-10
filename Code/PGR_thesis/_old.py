

# #%% Dirichlet RV, multivariate (generalized dimension)
#
# def _dirichlet_multi_check_parameters(alpha):
#     alpha = np.asarray(alpha)
#     if np.min(alpha) <= 0:
#         raise ValueError("All parameters must be greater than 0")
#     return alpha
#
#
# def _dirichlet_multi_check_input(x, alpha):
#     x, size = _multi_check_input_shape(x, alpha.shape)
#
#     if np.min(x) < 0:
#         raise ValueError("Each entry in 'x' must be greater than or equal "
#                          "to zero.")
#     if np.max(x) > 1:
#         raise ValueError("Each entry in 'x' must be smaller or equal one.")
#
#     if (np.abs(x.reshape(size, -1).sum(-1) - 1.0) > 1e-9).any():
#         raise ValueError("The input vector 'x' must lie within the normal "
#                          "simplex. but x.reshape(size, -1).sum(-1) = %s." % x.reshape(size, -1).sum(-1))
#
#     # if np.logical_and(x == 0, alpha < 1).any():       # TODO: need control over pdf inf values?
#     #     raise ValueError("Each entry in 'x' must be greater than zero if its "
#     #                      "alpha is less than one.")
#
#     return x, size
#
#
# class DirichletMultiGen(multi_rv_generic):
#
#     # TODO: normalized alpha and concentration?!?!?
#
#     def __init__(self, seed=None):
#         super(DirichletMultiGen, self).__init__(seed)
#         # self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)     # TODO: docstring?
#
#     def __call__(self, alpha, seed=None):
#         alpha = _dirichlet_multi_check_parameters(alpha)
#         return DirichletMultiFrozen(alpha, seed=seed)
#
#     # def _pdf_single(self, x, alpha):
#     #     log_pdf = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum(xlogy(alpha - 1, x))
#     #     return np.exp(log_pdf)
#     #
#     # def _pdf_vec(self, x, alpha):
#     #     return np.array([self._pdf_single(x_i, alpha) for x_i in x])
#
#     def pdf(self, x, alpha):
#         alpha = _dirichlet_multi_check_parameters(alpha)
#         x, size = _dirichlet_multi_check_input(x, alpha)
#         # if size is None:
#         #     return self._pdf_single(x, alpha)
#         # else:
#         #     return self._pdf_vec(x, alpha)
#         log_pdf = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum(xlogy(alpha - 1, x).reshape(size, -1), -1)
#         return np.exp(log_pdf)
#
#     def mean(self, alpha):
#         alpha = _dirichlet_multi_check_parameters(alpha)
#         return alpha / alpha.sum()
#
#     def cov(self, alpha):
#         alpha = _dirichlet_multi_check_parameters(alpha)
#         mean = self.mean(alpha)
#         return (diag_gen(mean) - outer_gen(mean, mean)) / (alpha.sum() + 1)
#
#     def mode(self, alpha):
#         alpha = _dirichlet_multi_check_parameters(alpha)
#         if np.min(alpha) <= 1:
#             raise NotImplementedError("Method currently supported for alpha > 1 only")
#             # TODO: complete with general formula
#         else:
#             return (alpha - 1) / (alpha.sum() - alpha.size)
#
#     def rvs(self, alpha, size=None, random_state=None):
#         alpha = _dirichlet_multi_check_parameters(alpha)
#         random_state = self._get_random_state(random_state)
#
#         if size is None:
#             return random_state.dirichlet(alpha.flatten()).reshape(alpha.shape)
#         else:
#             return random_state.dirichlet(alpha.flatten(), size).reshape((size,)+alpha.shape)
#
#
# dirichlet_multi = DirichletMultiGen()
#
#
# class DirichletMultiFrozen(multi_rv_frozen):
#     def __init__(self, alpha, seed=None):
#         self.alpha = alpha
#         self._dist = DirichletMultiGen(seed)
#
#     def pdf(self, x):
#         return self._dist.pdf(x, self.alpha)
#
#     def mean(self):
#         return self._dist.mean(self.alpha)
#
#     def cov(self):
#         return self._dist.cov(self.alpha)
#
#     def mode(self):
#         return self._dist.mode(self.alpha)
#
#     def rvs(self, size=None, random_state=None):
#         return self._dist.rvs(self.alpha, size, random_state)



# #%% Deterministic RV, univariate
#
# # TODO: depreciate, use DeterministicMultiGen?
#
# # def _deterministic_uni_check_parameters(val):
# #     val = np.asarray(val).squeeze()
# #     if val.size != 1:
# #         raise TypeError('Parameter must be singular.')
# #     return val
# #
# #
# # def _deterministic_uni_check_input(val, x):
# #     val = _deterministic_uni_check_parameters(val)
# #     x = np.asarray(x).squeeze()
# #     if x.shape != val.shape:
# #         raise TypeError(f'Input must be singular.')
# #     return x
#
# class DeterministicUniGen(rv_continuous):
#
#     def _cdf(self, x, *args):
#         return np.where(x < 0, 0., 1.)
#
#     def _pdf(self, x, *args):
#         return np.where(x != 0, 0., np.inf)
#
#     def _stats(self, *args, **kwds):
#         return 0., 0., 0., 0.
#
#     def _rvs(self, *args):
#         return np.zeros(self._size)
#
#     def median(self, *args, **kwds):
#         args, loc, scale = self._parse_args(*args, **kwds)
#         return float(loc)
#
#     def mode(self, *args, **kwds):  # TODO: cannot be accessed through Frozen RV, no method for rv_frozen
#         args, loc, scale = self._parse_args(*args, **kwds)
#         # loc, scale = map(np.asarray, (loc, scale))
#         return float(loc)
#
# deterministic_uni = DeterministicUniGen(name='deterministic')  # TODO: block non-singular inputs?