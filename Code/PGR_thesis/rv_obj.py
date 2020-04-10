"""
Random element objects
"""

import numpy as np
from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
from scipy.special import gammaln, xlogy
import warnings

from util.util import outer_gen, diag_gen


def _multi_check_input_shape(x, shape):
    """Checks input shape for RV cdf/pdf calls"""
    x = np.asarray(x)

    if x.shape == shape:
        size = None
    elif x.ndim == len(shape) + 1 and x.shape[1:] == shape:
        size = x.shape[0]
    else:
        raise TypeError(f"Input 'x' shape must be equal to {shape} or (size,)+{shape}.")

    return x, size


#%% Deterministic RV, multivariate

# TODO: make deterministic a rv_discrete subclass?

def _deterministic_multi_check_parameters(val):
    return np.asarray(val)


def _deterministic_multi_check_input(x, val):
    x, size = _multi_check_input_shape(x, val.shape)
    return x, size


class DeterministicMultiGen(multi_rv_generic):

    def __init__(self, seed=None):
        super(DeterministicMultiGen, self).__init__(seed)
        # self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)     # TODO: docstring?

    # TODO: lots of redundant parameter check calls, even when frozen, like scipy.stats

    def __call__(self, val):
        return DeterministicMultiFrozen(val)


    # def _cdf_single(self, x, val):
    #     return 1. if np.all(x >= val) else 0.
    #
    # def _cdf_vec(self, x, val):
    #     return np.array([self._cdf_single(x_i, val) for x_i in x])
    #     # np.vectorize(self._cdf_single, signature='(n,...,m),(n,...,m)->()')(x, val)  # TODO: np.vectorize?

    @staticmethod
    def cdf(x, val):
        val = _deterministic_multi_check_parameters(val)
        x, size = _deterministic_multi_check_input(x, val)
        # if size is None:
        #     return self._cdf_single(x, val)
        # else:
        #     return self._cdf_vec(x, val)
        return np.where(np.all(x.reshape(-1, val.size) >= val.flatten(), axis=-1).squeeze(), 1., 0.)

    # def _pdf_single(self, x, val):
    #     return np.inf if np.all(x == val) else 0.
    #
    # def _pdf_vec(self, x, val):
    #     return np.array([self._pdf_single(x_i, val) for x_i in x])

    @staticmethod
    def pdf(x, val):
        val = _deterministic_multi_check_parameters(val)
        x, size = _deterministic_multi_check_input(x, val)
        # if size is None:
        #     return self._pdf_single(x, val)
        # else:
        #     return self._pdf_vec(x, val)
        return np.where(np.all(x.reshape(-1, val.size) == val.flatten(), axis=-1).squeeze(), np.inf, 0.)

    @staticmethod
    def mean(val):
        val = _deterministic_multi_check_parameters(val)
        return val

    @staticmethod
    def cov(val):
        val = _deterministic_multi_check_parameters(val)
        return np.zeros(2*val.shape)

    @staticmethod
    def mode(val):
        val = _deterministic_multi_check_parameters(val)
        return val

    @staticmethod
    def rvs(val, size=None):
        val = _deterministic_multi_check_parameters(val)
        if size is None:
            return val
        else:
            return np.broadcast_to(val, (size,) + val.shape)


deterministic_multi = DeterministicMultiGen()


class DeterministicMultiFrozen(multi_rv_frozen):
    def __init__(self, val):
        self.val = _deterministic_multi_check_parameters(val)
        self._dist = DeterministicMultiGen()

    def cdf(self, x):
        return self._dist.cdf(x, self.val)

    def pdf(self, x):
        return self._dist.pdf(x, self.val)

    @property
    def mean(self):
        return self._dist.mean(self.val)

    @property
    def cov(self):
        return self._dist.cov(self.val)

    @property
    def mode(self):
        return self._dist.mode(self.val)

    def rvs(self, size=None):
        return self._dist.rvs(self.val, size)


#%% Dirichlet RV, multivariate (generalized dimension)

def _dirichlet_multi_check_parameters(alpha_0, mean):
    alpha_0 = np.asarray(alpha_0)
    if alpha_0.size > 1 or alpha_0 <= 0:
        raise ValueError("Concentration parameter must be a positive scalar")
    mean = np.asarray(mean)
    if np.min(mean) <= 0:
        raise ValueError("All parameters must be greater than 0")
    return alpha_0, mean


def _dirichlet_multi_check_input(x, alpha_0, mean):
    x, size = _multi_check_input_shape(x, mean.shape)

    if np.min(x) < 0:
        raise ValueError("Each entry in 'x' must be greater than or equal "
                         "to zero.")
    if np.max(x) > 1:
        raise ValueError("Each entry in 'x' must be smaller or equal one.")

    if (np.abs(x.reshape(size, -1).sum(-1) - 1.0) > 1e-9).any():
        raise ValueError("The input vector 'x' must lie within the normal "
                         "simplex. but x.reshape(size, -1).sum(-1) = %s." % x.reshape(size, -1).sum(-1))

    if np.logical_and(x == 0, mean < 1 / alpha_0).any():
        raise ValueError("Each entry in 'x' must be greater than zero if its "
                         "mean is less than 1 / alpha_0.")

    return x, size


class DirichletMultiGen(multi_rv_generic):

    def __init__(self, seed=None):
        super(DirichletMultiGen, self).__init__(seed)
        # self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)     # TODO: docstring?

    def __call__(self, alpha_0, mean, seed=None):
        return DirichletMultiFrozen(alpha_0, mean, seed=seed)

    # def _pdf_single(self, x, alpha):
    #     log_pdf = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum(xlogy(alpha - 1, x))
    #     return np.exp(log_pdf)
    #
    # def _pdf_vec(self, x, alpha):
    #     return np.array([self._pdf_single(x_i, alpha) for x_i in x])

    @staticmethod
    def pdf(x, alpha_0, mean):
        alpha_0, mean = _dirichlet_multi_check_parameters(alpha_0, mean)
        x, size = _dirichlet_multi_check_input(x, alpha_0, mean)
        # if size is None:
        #     return self._pdf_single(x, alpha)
        # else:
        #     return self._pdf_vec(x, alpha)
        log_pdf = gammaln(np.sum(alpha_0 * mean)) - np.sum(gammaln(alpha_0 * mean)) \
                  + np.sum(xlogy(alpha_0 * mean - 1, x).reshape(size, -1), -1)
        return np.exp(log_pdf)

    # def mean(self, alpha_0, mean):
    #     _, mean = _dirichlet_multi_check_parameters(alpha_0, mean)
    #     return mean

    @staticmethod
    def cov(alpha_0, mean):
        alpha_0, mean = _dirichlet_multi_check_parameters(alpha_0, mean)
        return (diag_gen(mean) - outer_gen(mean, mean)) / (alpha_0 + 1)

    @staticmethod
    def mode(alpha_0, mean):
        alpha_0, mean = _dirichlet_multi_check_parameters(alpha_0, mean)
        if np.min(mean) <= 1 / alpha_0:
            warnings.warn("No output. Method currently supported for mean > 1/alpha_0 only")
            # TODO: complete with general formula
            return None
        else:
            return (mean - 1 / alpha_0) / (1 - mean.size / alpha_0)

    def rvs(self, alpha_0, mean, size=None, random_state=None):
        alpha_0, mean = _dirichlet_multi_check_parameters(alpha_0, mean)
        random_state = self._get_random_state(random_state)

        if size is None:
            return random_state.dirichlet(alpha_0 * mean.flatten()).reshape(mean.shape)
        else:
            return random_state.dirichlet(alpha_0 * mean.flatten(), size).reshape((size,)+mean.shape)


dirichlet_multi = DirichletMultiGen()


class DirichletMultiFrozen(multi_rv_frozen):
    def __init__(self, alpha_0, mean, seed=None):
        self.alpha_0, self.mean = _dirichlet_multi_check_parameters(alpha_0, mean)
        self._dist = DirichletMultiGen(seed)

    def pdf(self, x):
        return self._dist.pdf(x, self.alpha_0, self.mean)

    @property
    def cov(self):
        return self._dist.cov(self.alpha_0, self.mean)

    @property
    def mode(self):
        return self._dist.mode(self.alpha_0, self.mean)

    def rvs(self, size=None, random_state=None):
        return self._dist.rvs(self.alpha_0, self.mean, size, random_state)


