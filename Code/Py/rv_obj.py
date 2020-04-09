import numpy as np
from scipy import stats
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
from scipy.special import gammaln, xlogy


#%% Deterministic RV, multivariate

# TODO: make deterministic a rv_discrete subclass?

def _deterministic_multi_check_parameters(val):
    return np.asarray(val)


def _deterministic_multi_check_input(x, val):
    x = np.asarray(x)

    if x.shape == val.shape:
        size = None
    elif x.ndim == val.ndim + 1 and x.shape[1:] == val.shape:
        size = x.shape[0]
    else:
        raise TypeError(f'Input shape must be equal to {val.shape} or (i,)+{val.shape}.')

    return x, size


class DeterministicMultiGen(multi_rv_generic):

    def __init__(self, seed=None):
        super(DeterministicMultiGen, self).__init__(seed)
        # self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)     # TODO: docstring?

    # TODO: lots of redundant parameter check calls, even when frozen, like scipy.stats

    def __call__(self, val):
        val = _deterministic_multi_check_parameters(val)
        return DeterministicMultiFrozen(val)

    def _cdf_single(self, x, val):      # TODO: make these into static methods!?
        return 1. if np.all(x >= val) else 0.

    def _cdf_vec(self, x, val):
        return np.array([self._cdf_single(x_i, val) for x_i in x])     # TODO: np.vectorize?
        # np.vectorize(self._cdf_single, signature='(n,...,m),(n,...,m)->()')(x, val)  # TODO: BROKEN SIGS!
        # return np.where(np.all(x >= val, axis=tuple(range(1, x.ndim))), 1., 0.)

    def cdf(self, x, val):
        val = _deterministic_multi_check_parameters(val)
        x, size = _deterministic_multi_check_input(x, val)
        if size is None:
            return self._cdf_single(x, val)
        else:
            return self._cdf_vec(x, val)

    def _pdf_single(self, x, val):
        return np.inf if np.all(x == val) else 0.

    def _pdf_vec(self, x, val):
        return np.array([self._pdf_single(x_i, val) for x_i in x])

    def pdf(self, x, val):
        val = _deterministic_multi_check_parameters(val)
        x, size = _deterministic_multi_check_input(x, val)
        if size is None:
            return self._pdf_single(x, val)
        else:
            return self._pdf_vec(x, val)

    def mean(self, val):
        val = _deterministic_multi_check_parameters(val)
        return val

    def var(self, val):
        val = _deterministic_multi_check_parameters(val)
        return np.zeros(2*val.shape)

    def median(self, val):
        val = _deterministic_multi_check_parameters(val)
        return val

    def mode(self, val):
        val = _deterministic_multi_check_parameters(val)
        return val

    def rvs(self, val, size=None):
        val = _deterministic_multi_check_parameters(val)
        if size is None:
            return val
        elif isinstance(size, (np.int, np.float)) and size >= 1:
            size = int(size)
            return np.broadcast_to(val[np.newaxis], (size,)+val.shape)
        else:
            raise TypeError("Input 'size' must be a positive integer.")

deterministic_multi = DeterministicMultiGen()


class DeterministicMultiFrozen(multi_rv_frozen):
    def __init__(self, val):
        self.val = val
        self._dist = DeterministicMultiGen()

    def cdf(self, x):
        return self._dist.cdf(x, self.val)

    def pdf(self, x):
        return self._dist.pdf(x, self.val)

    def mean(self):
        return self._dist.mean(self.val)

    def var(self):
        return self._dist.var(self.val)

    def median(self):
        return self._dist.median(self.val)

    def mode(self):
        return self._dist.mode(self.val)

    def rvs(self, size=None):
        return self._dist.rvs(self.val, size)


#%% Dirichlet RV, multivariate (generalized dimension)

def _dirichlet_multi_check_parameters(alpha):
    alpha = np.asarray(alpha)

    if np.min(alpha) <= 0:
        raise ValueError("All parameters must be greater than 0")

    return alpha


def _dirichlet_multi_check_input(alpha, x):
    x = np.asarray(x)

    if x.shape == alpha.shape:
        size = None
    elif x.ndim == alpha.ndim + 1 and x.shape[1:] == alpha.shape:
        size = x.shape[0]
    else:
        raise TypeError(f'Input shape must be equal to {alpha.shape} or (i,)+{alpha.shape}.')

    if np.min(x) < 0:
        raise ValueError("Each entry in 'x' must be greater than or equal "
                         "to zero.")

    if np.max(x) > 1:
        raise ValueError("Each entry in 'x' must be smaller or equal one.")

    # Check x_i > 0 or alpha_i > 1
    xeq0 = (x == 0)
    alphalt1 = (alpha < 1)
    if size is not None:
        alphalt1 = np.broadcast_to(alphalt1[np.newaxis], (size,) + alpha.shape)
    if np.logical_and(xeq0, alphalt1).any():
        raise ValueError("Each entry in 'x' must be greater than zero if its "
                         "alpha is less than one.")

    if (np.abs(x.reshape(size, -1).sum(-1) - 1.0) > 1e-9).any():
        raise ValueError("The input vector 'x' must lie within the normal "
                         "simplex. but x.reshape(size, -1).sum(-1) = %s." % x.reshape(size, -1).sum(-1))

    return x, size


class DirichletMultiGen(multi_rv_generic):

    # TODO: normalized alpha!?!?

    def __init__(self, seed=None):
        super(DirichletMultiGen, self).__init__(seed)
        # self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)     # TODO: docstring?

    def __call__(self, alpha, seed=None):
        alpha = _dirichlet_multi_check_parameters(alpha)
        return DirichletMultiFrozen(alpha, seed=seed)

    def _pdf_single(self, x, alpha):
        log_pdf = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum(xlogy(alpha - 1, x))
        return np.exp(log_pdf)

    def _pdf_vec(self, x, alpha):
        return np.array([self._pdf_single(x_i, alpha) for x_i in x])

    def pdf(self, x, alpha):        # TODO: efficient broadcasting like in _multivariate?
        alpha = _dirichlet_multi_check_parameters(alpha)
        x, size = _dirichlet_multi_check_input(alpha, x)
        if size is None:
            return self._pdf_single(x, alpha)
        else:
            return self._pdf_vec(x, alpha)

    def mean(self, alpha):
        alpha = _dirichlet_multi_check_parameters(alpha)
        return alpha / alpha.sum()

    def var(self, alpha):
        alpha = _dirichlet_multi_check_parameters(alpha)
        return np.zeros(2*alpha.shape)

    def median(self, alpha):
        alpha = _dirichlet_multi_check_parameters(alpha)
        return alpha

    def mode(self, alpha):
        alpha = _dirichlet_multi_check_parameters(alpha)
        return alpha

    def rvs(self, alpha, size=None, random_state=None):
        alpha = _dirichlet_multi_check_parameters(alpha)
        random_state = self._get_random_state(random_state)

        if size is None:
            return random_state.dirichlet(alpha.flatten()).reshape(alpha.shape)
        else:
            return random_state.dirichlet(alpha.flatten(), size).reshape((size,)+alpha.shape)


dirichlet_multi = DirichletMultiGen()


class DirichletMultiFrozen(multi_rv_frozen):
    def __init__(self, alpha, seed=None):
        self.alpha = alpha
        self._dist = DirichletMultiGen(seed)

    def pdf(self, x):
        return self._dist.pdf(x, self.alpha)

    def mean(self):
        return self._dist.mean(self.alpha)

    def var(self):
        return self._dist.var(self.alpha)

    def median(self):
        return self._dist.median(self.alpha)

    def mode(self):
        return self._dist.mode(self.alpha)

    def rvs(self, size=None, random_state=None):
        return self._dist.rvs(self.alpha, size, random_state)




    # # %% Deterministic RV, univariate
    #
    # # TODO: depreciate in favor of multivariate?
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