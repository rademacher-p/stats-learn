"""
Random element objects
"""

# TODO: docstrings?

import numpy as np
from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
from scipy.special import gammaln, xlogy
import warnings

from util.util import outer_gen, diag_gen


def _multi_check_input_shape(x, shape):     # TODO: generalize for n-dim?
    """Checks input shape for RV cdf/pdf calls"""
    x = np.asarray(x)

    if x.shape == shape:
        size = None
    elif x.ndim == len(shape) + 1 and x.shape[1:] == shape:
        size = len(x)
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
    x, _ = _multi_check_input_shape(x, shape)
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("Input 'x' must be of numeric type.")
    return x


class DeterministicPGR(object):
    def __init__(self, val):
        self.val = _deterministic_multi_check_parameters(val)

    def cdf(self, x):
        x = _deterministic_multi_check_input(x, self.val.shape)
        return np.where(np.all(x.reshape(-1, self.val.size) >= self.val.flatten(), axis=-1).squeeze(), 1., 0.)

    def pdf(self, x):
        x = _deterministic_multi_check_input(x, self.val.shape)
        return np.where(np.all(x.reshape(-1, self.val.size) == self.val.flatten(), axis=-1).squeeze(), np.inf, 0.)

    @property
    def mean(self):
        return self.val

    @property
    def cov(self):
        return np.zeros(2 * self.val.shape)

    @property
    def mode(self):
        return self.val

    def rvs(self, size=None):
        if size is None:
            return self.val
        else:
            return np.broadcast_to(self.val, (size,) + self.val.shape)


#%% Dirichlet RV, multivariate (generalized dimension)

def _dirichlet_check_alpha_0(alpha_0):
    alpha_0 = np.asarray(alpha_0)
    if alpha_0.size > 1 or alpha_0 <= 0:
        raise ValueError("Concentration parameter must be a positive scalar")
    return alpha_0


def _dirichlet_check_mean(mean):
    mean = np.asarray(mean)
    if np.min(mean) < 0:
        raise ValueError("Each entry in 'mean' must be greater than or equal "
                         "to zero.")
    if np.abs(mean.sum() - 1.0) > 1e-9:
        raise ValueError("The input 'mean' must lie within the normal "
                         "simplex. but mean.sum() = %s." % mean.sum())
    return mean


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


class DirichletPGR(multi_rv_generic):
    def __init__(self, alpha_0, mean, seed=None):
        super().__init__(seed)
        self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
        self._mean = _dirichlet_check_mean(mean)
        self._update_attr()

    @property
    def alpha_0(self):
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
        self._update_attr()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = _dirichlet_check_mean(mean)
        self._update_attr()

    def _update_attr(self):
        self._cov = (diag_gen(self.mean) - outer_gen(self.mean, self.mean)) / (self.alpha_0 + 1)

        if np.min(self.mean) <= 1 / self.alpha_0:
            # TODO: complete with general formula
            self._mode = None
        else:
            self._mode = (self.mean - 1 / self.alpha_0) / (1 - self.mean.size / self.alpha_0)

        self._beta_inv = gammaln(np.sum(self.alpha_0 * self.mean)) - np.sum(gammaln(self.alpha_0 * self.mean))

    @property
    def cov(self):
        return self._cov

    @property
    def mode(self):
        if self._mode is None:
            # warnings.warn("No output. Method currently supported for mean > 1/alpha_0 only")
            raise ValueError("No output. Method currently supported for mean > 1/alpha_0 only")
        return self._mode

    def pdf(self, x):
        x, size = _dirichlet_multi_check_input(x, self.alpha_0, self.mean)

        log_pdf = self._beta_inv + np.sum(xlogy(self.alpha_0 * self.mean - 1, x).reshape(size, -1), -1)
        return np.exp(log_pdf)

    def rvs(self, size=None, random_state=None):
        random_state = self._get_random_state(random_state)
        if size is None:
            return random_state.dirichlet(self.alpha_0 * self.mean.flatten()).reshape(self.mean.shape)
        else:
            return random_state.dirichlet(self.alpha_0 * self.mean.flatten(), size).reshape((size,) + self.mean.shape)


#%% Discrete RV, multivariate (generalized)

# TODO: modify for non-scalar elements?

# TODO: use structured array to combine support and pmf?

def _discrete_check_parameters(supp, p):
    supp = np.asarray(supp)
    p = np.asarray(p)

    set_shape, data_shape = supp.shape[:p.ndim], supp.shape[p.ndim:]
    if set_shape != p.shape:
        raise ValueError("Leading shape values of 'supp' must equal the shape of 'pmf'.")

    # if pmf.shape != support.shape:
    #     raise TypeError("Input 'pmf' must have the same shape as 'support'.")

    # if len(data_shape) == 0:        # TODO: check?
    #     supp_flat = supp.flatten()
    # else:
    #     supp_flat = supp.reshape((np.prod(set_shape), np.prod(data_shape)))
    supp_flat = supp.reshape((np.prod(set_shape), -1))

    # if support.size != np.unique(support).size:
    if len(supp_flat) != len(np.unique(supp_flat, axis=0)):
        raise ValueError("Input 'supp' must have unique values")

    if np.min(p) < 0:
        raise ValueError("Each entry in 'pmf' must be greater than or equal "
                         "to zero.")
    if np.abs(p.sum() - 1.0) > 1e-9:
        raise ValueError("The input 'pmf' must lie within the normal "
                         "simplex. but pmf.sum() = %s." % p.sum())

    return supp, p


def _discrete_multi_check_input(x, supp):
    x, size = _multi_check_input_shape(x, supp.shape)
    if not np.isin(x, supp).all():
        raise ValueError("Elements of input 'x' must be in the support set %s." % supp)

    return x, size      # TODO: remove size?


class FinitePGR(multi_rv_generic):
    def __init__(self, supp, p, seed=None):
        super().__init__(seed)
        self._supp, self._p = _discrete_check_parameters(supp, p)
        self._update_attr()

    @property
    def supp(self):
        return self._supp

    @supp.setter
    def supp(self, supp):
        self._supp, self._p = _discrete_check_parameters(supp, self.p)
        self._update_attr()

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._supp, self._p = _discrete_check_parameters(self.supp, p)
        self._update_attr()

    def _update_attr(self):
        # _, self._data_shape = self.supp.shape[:self.p.ndim], self.supp.shape[self.p.ndim:]
        self._data_shape = self.supp.shape[self.p.ndim:]
        self._supp_flat, self._p_flat = self.supp.reshape((np.prod(self.p.shape), -1)), self.p.flatten()

        # if all([np.issubdtype(support.dtype[i], np.number) for i in range(len(support.dtype))]):
        if np.issubdtype(self.supp.dtype, np.number):
            # self._mean = (self.supp * np.broadcast_to(self.p, self.supp.shape)).sum()
            mean_flat = (self._p_flat[:, np.newaxis] * self._supp_flat).sum(axis=0)
            self._mean = mean_flat.reshape(self._data_shape)
            ctr_flat = self._supp_flat - mean_flat
            outer_flat = (ctr_flat.reshape(self.p.size, 1, -1) * ctr_flat[..., np.newaxis]).reshape(self.p.size, -1)
            self._cov = (self._p_flat[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self._data_shape)
        else:
            self._mean = None
            self._cov = None

        self._mode = self.supp.flatten()[np.argmax(self.p)]

    @property
    def mean(self):
        if self._mean is None:
            # warnings.warn("Method only supported for numeric 'supp'.")
            raise TypeError("Method only supported for numeric 'supp'.")
        return self._mean

    @property
    def cov(self):
        if self._cov is None:
            # warnings.warn("Method only supported for numeric 'supp'.")
            raise TypeError("Method only supported for numeric 'supp'.")
        return self._cov

    @property
    def mode(self):
        return self._mode

    def pmf(self, x):
        x, _ = _discrete_multi_check_input(x, self.supp)
        return self.p[x == self.supp]

    def rvs(self, size=None, random_state=None):
        random_state = self._get_random_state(random_state)
        i = random_state.choice(self.p.size, size, p=self._p_flat)
        if size is None:
            return self._supp_flat[i].reshape(self._data_shape)
        else:
            return self._supp_flat[i].reshape((size,) + self._data_shape)
        # return random_state.choice(self.supp.flatten(), size, p=self.p.flatten())

