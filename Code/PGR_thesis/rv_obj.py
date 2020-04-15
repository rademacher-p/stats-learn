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

def _deterministic_multi_check_parameters(val):
    val = np.asarray(val)
    if not np.issubdtype(val.dtype, np.number):
        raise TypeError("Input 'val' must be of numeric type.")     # TODO: numeric/continuous only? change name?
    return np.asarray(val)


def _deterministic_multi_check_input(x, shape):
    x, size = _multi_check_input_shape(x, shape)
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("Input 'x' must be of numeric type.")
    return x, size


class DeterministicMultiGen(multi_rv_generic):

    def __init__(self, seed=None):
        super(DeterministicMultiGen, self).__init__(seed)
        # self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)     # TODO: docstring?

    def __call__(self, val):
        return DeterministicMultiFrozen(val)

    # def _cdf_single(self, x, val):
    #     return 1. if np.all(x >= val) else 0.
    #
    # def _cdf_vec(self, x, val):
    #     return np.array([self._cdf_single(x_i, val) for x_i in x])
    #     # np.vectorize(self._cdf_single, signature='(n,...,m),(n,...,m)->()')(x, val)  # TODO: np.vectorize?

    @staticmethod
    def _cdf(x, val):
        x, _ = _deterministic_multi_check_input(x, val.shape)
        return np.where(np.all(x.reshape(-1, val.size) >= val.flatten(), axis=-1).squeeze(), 1., 0.)

    @staticmethod
    def cdf(x, val):
        val = _deterministic_multi_check_parameters(val)

        # if size is None:
        #     return self._cdf_single(x, val)
        # else:
        #     return self._cdf_vec(x, val)

        return DeterministicMultiGen._cdf(x, val)

    # def _pdf_single(self, x, val):
    #     return np.inf if np.all(x == val) else 0.
    #
    # def _pdf_vec(self, x, val):
    #     return np.array([self._pdf_single(x_i, val) for x_i in x])

    @staticmethod
    def _pdf(x, val):
        x, _ = _deterministic_multi_check_input(x, val.shape)
        return np.where(np.all(x.reshape(-1, val.size) == val.flatten(), axis=-1).squeeze(), np.inf, 0.)

    @staticmethod
    def pdf(x, val):
        val = _deterministic_multi_check_parameters(val)

        # if size is None:
        #     return self._pdf_single(x, val)
        # else:
        #     return self._pdf_vec(x, val)

        return DeterministicMultiGen._pdf(x, val)

    @staticmethod
    def _mean(val):
        return val

    @staticmethod
    def mean(val):
        val = _deterministic_multi_check_parameters(val)
        return DeterministicMultiGen._mean(val)

    @staticmethod
    def _cov(val):
        return np.zeros(2 * val.shape)

    @staticmethod
    def cov(val):
        val = _deterministic_multi_check_parameters(val)
        return DeterministicMultiGen._cov(val)

    @staticmethod
    def _mode(val):
        return val

    @staticmethod
    def mode(val):
        val = _deterministic_multi_check_parameters(val)
        return DeterministicMultiGen._mode(val)

    @staticmethod
    def _rvs(val, size=None):
        if size is None:
            return val
        else:
            return np.broadcast_to(val, (size,) + val.shape)

    @staticmethod
    def rvs(val, size=None):
        val = _deterministic_multi_check_parameters(val)
        return DeterministicMultiGen._rvs(val, size)


deterministic_multi = DeterministicMultiGen()


class DeterministicMultiFrozen(multi_rv_frozen):
    def __init__(self, val):
        self.val = _deterministic_multi_check_parameters(val)
        self._dist = DeterministicMultiGen()

    def cdf(self, x):
        return self._dist._cdf(x, self.val)

    def pdf(self, x):
        return self._dist._pdf(x, self.val)

    @property
    def mean(self):
        return self._dist._mean(self.val)

    @property
    def cov(self):
        return self._dist._cov(self.val)

    @property
    def mode(self):
        return self._dist._mode(self.val)

    def rvs(self, size=None):
        return self._dist._rvs(self.val, size)


#%% Dirichlet RV, multivariate (generalized dimension)

def _dirichlet_multi_check_parameters(alpha_0, mean):
    alpha_0 = np.asarray(alpha_0)
    if alpha_0.size > 1 or alpha_0 <= 0:
        raise ValueError("Concentration parameter must be a positive scalar")

    mean = np.asarray(mean)
    if np.min(mean) < 0:
        raise ValueError("Each entry in 'mean' must be greater than or equal "
                         "to zero.")

    if np.abs(mean.sum() - 1.0) > 1e-9:
        raise ValueError("The input 'mean' must lie within the normal "
                         "simplex. but mean.sum() = %s." % mean.sum())

    return alpha_0, mean


def _dirichlet_multi_check_input(x, alpha_0, mean):
    x, size = _multi_check_input_shape(x, mean.shape)

    if np.min(x) < 0:
        raise ValueError("Each entry in 'x' must be greater than or equal "
                         "to zero.")

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

    def __call__(self, alpha_0, mean, seed=None):
        return DirichletMultiFrozen(alpha_0, mean, seed=seed)

    # def _pdf_single(self, x, alpha):
    #     log_pdf = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum(xlogy(alpha - 1, x))
    #     return np.exp(log_pdf)
    #
    # def _pdf_vec(self, x, alpha):
    #     return np.array([self._pdf_single(x_i, alpha) for x_i in x])

    @staticmethod
    def _pdf(x, alpha_0, mean):
        x, size = _dirichlet_multi_check_input(x, alpha_0, mean)

        log_pdf = gammaln(np.sum(alpha_0 * mean)) - np.sum(gammaln(alpha_0 * mean)) \
                  + np.sum(xlogy(alpha_0 * mean - 1, x).reshape(size, -1), -1)
        return np.exp(log_pdf)

    @staticmethod
    def pdf(x, alpha_0, mean):
        alpha_0, mean = _dirichlet_multi_check_parameters(alpha_0, mean)

        # if size is None:
        #     return self._pdf_single(x, alpha)
        # else:
        #     return self._pdf_vec(x, alpha)

        return DirichletMultiGen._pdf(x, alpha_0, mean)

    @staticmethod
    def _cov(alpha_0, mean):
        return (diag_gen(mean) - outer_gen(mean, mean)) / (alpha_0 + 1)

    @staticmethod
    def cov(alpha_0, mean):
        alpha_0, mean = _dirichlet_multi_check_parameters(alpha_0, mean)
        return DirichletMultiGen._cov(alpha_0, mean)

    @staticmethod
    def _mode(alpha_0, mean):
        if np.min(mean) <= 1 / alpha_0:
            warnings.warn("No output. Method currently supported for mean > 1/alpha_0 only")
            # TODO: complete with general formula
            return None
        else:
            return (mean - 1 / alpha_0) / (1 - mean.size / alpha_0)

    @staticmethod
    def mode(alpha_0, mean):
        alpha_0, mean = _dirichlet_multi_check_parameters(alpha_0, mean)
        return DirichletMultiGen._mode(alpha_0, mean)

    def _rvs(self, alpha_0, mean, size=None, random_state=None):
        random_state = self._get_random_state(random_state)

        if size is None:
            return random_state.dirichlet(alpha_0 * mean.flatten()).reshape(mean.shape)
        else:
            return random_state.dirichlet(alpha_0 * mean.flatten(), size).reshape((size,)+mean.shape)

    def rvs(self, alpha_0, mean, size=None, random_state=None):
        alpha_0, mean = _dirichlet_multi_check_parameters(alpha_0, mean)
        return self._rvs(alpha_0, mean, size, random_state)


dirichlet_multi = DirichletMultiGen()


class DirichletMultiFrozen(multi_rv_frozen):
    def __init__(self, alpha_0, mean, seed=None):
        self.alpha_0, self.mean = _dirichlet_multi_check_parameters(alpha_0, mean)
        self._dist = DirichletMultiGen(seed)

    def pdf(self, x):
        return self._dist._pdf(x, self.alpha_0, self.mean)

    @property
    def cov(self):
        return self._dist._cov(self.alpha_0, self.mean)

    @property
    def mode(self):
        return self._dist._mode(self.alpha_0, self.mean)

    def rvs(self, size=None, random_state=None):
        return self._dist._rvs(self.alpha_0, self.mean, size, random_state)


#%% Discrete RV, multivariate (generalized)

# TODO: modify for non-scalar elements?
# TODO: use structured array to combine support and pmf?
# TODO: pmf field/method naming conflict?

def _discrete_multi_check_parameters(support, pmf):
    support = np.asarray(support)
    if support.size != np.unique(support).size:
        raise ValueError("Input 'support' must have unique values")

    # if np.issubdtype(support.dtype, np.number):
    if all([np.issubdtype(support.dtype[i], np.number) for i in range(len(support.dtype))]):
        is_numeric = True
    else:
        is_numeric = False

    pmf = np.asarray(pmf)
    if pmf.shape != support.shape:
        raise TypeError("Input 'pmf' must have the same shape as 'support'.")

    if np.min(pmf) < 0:
        raise ValueError("Each entry in 'pmf' must be greater than or equal "
                         "to zero.")
    if np.abs(pmf.sum() - 1.0) > 1e-9:
        raise ValueError("The input 'pmf' must lie within the normal "
                         "simplex. but pmf.sum() = %s." % pmf.sum())

    return support, pmf, is_numeric


def _discrete_multi_check_input(x, support):
    x, size = _multi_check_input_shape(x, support.shape)
    if not np.isin(x, support).all():
        raise ValueError("Elements of input 'x' must be in the support set %s." % support)

    return x, size


class DiscreteMultiGen(multi_rv_generic):

    def __init__(self, seed=None):
        super(DiscreteMultiGen, self).__init__(seed)

    def __call__(self, support, pmf, seed=None):
        return DiscreteMultiFrozen(support, pmf, seed=seed)

    @staticmethod
    def _pmf(x, support, pmf):
        x, size = _discrete_multi_check_input(x, support)
        return pmf[x == support]

    @staticmethod
    def pmf(x, support, pmf):
        support, pmf, _ = _discrete_multi_check_parameters(support, pmf)
        return DiscreteMultiGen._pmf(x, support, pmf)

    @staticmethod
    def _mean(support, pmf, is_numeric):
        if is_numeric:
            return (support * pmf).sum()
        else:
            raise TypeError("Method only supported for numeric 'support'.")

    @staticmethod
    def mean(support, pmf):
        support, pmf, is_numeric = _discrete_multi_check_parameters(support, pmf)
        return DiscreteMultiGen._mean(support, pmf, is_numeric)

    @staticmethod
    def _cov(support, pmf, is_numeric):
        if is_numeric:
            return ((support - DiscreteMultiGen._mean(support, pmf))*2 * pmf).sum()
        else:
            raise TypeError("Method only supported for numeric 'support'.")

    @staticmethod
    def cov(support, pmf):
        support, pmf, is_numeric = _discrete_multi_check_parameters(support, pmf)
        return DiscreteMultiGen._cov(support, pmf, is_numeric)

    @staticmethod
    def _mode(support, pmf):
        return support[np.argmax(pmf)]

    @staticmethod
    def mode(support, pmf):
        support, pmf, _ = _discrete_multi_check_parameters(support, pmf)
        return DiscreteMultiGen._mode(support, pmf)

    def _rvs(self, support, pmf, size=None, random_state=None):
        random_state = self._get_random_state(random_state)
        return random_state.choice(support, size, p=pmf)

    def rvs(self, support, pmf, size=None, random_state=None):
        support, pmf, _ = _discrete_multi_check_parameters(support, pmf)
        return self._rvs(support, pmf, size, random_state)


discrete_multi = DiscreteMultiGen()


class DiscreteMultiFrozen(multi_rv_frozen):
    def __init__(self, support, pmf, seed=None):
        self.support, self.pmf, self.is_numeric = _discrete_multi_check_parameters(support, pmf)
        self._dist = DiscreteMultiGen(seed)

    def pmf(self, x):
        return self._dist._pmf(x, self.support, self.pmf)

    @property
    def mean(self):
        return self._dist._mean(self.support, self.pmf, self.is_numeric)

    @property
    def cov(self):
        return self._dist._cov(self.support, self.pmf, self.is_numeric)

    @property
    def mode(self):
        return self._dist._mode(self.support, self.pmf)

    def rvs(self, size=None, random_state=None):
        return self._dist._rvs(self.support, self.pmf, size, random_state)


