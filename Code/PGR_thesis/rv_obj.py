"""
Random element objects
"""

# TODO: docstrings?

import numpy as np
from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
from scipy.special import gammaln, xlogy
import warnings

from util.util import outer_gen, diag_gen


def _check_data_shape(x, shape):     # TODO: CHECK USES FOR N-DIM GENERALIZATION
    """Checks input shape for RV cdf/pdf calls"""

    x = np.asarray(x)
    if x.shape == shape:
        size = None
    elif len(shape) == 0:
        size = x.shape
    elif x.shape[-len(shape):] == shape:
        size = x.shape[:-len(shape)]
    else:
        raise TypeError("Trailing dimensions of 'shape' must be equal to the shape of 'x'.")

    return x, size


class BaseRE(multi_rv_generic):
    def __init__(self, *args, seed=None):
        super().__init__(seed)
        self._check_inputs(*args)
        self._update_attr()

    def _check_inputs(self, *args):       # TODO: kwargs?? placeholder!
        return args

    def _update_attr(self):
        self._mode = None
        self._mean = None
        self._cov = None

        # # _, self._data_shape = self.supp.shape[:self.p.ndim], self.supp.shape[self.p.ndim:]
        # self._data_shape = self.supp.shape[self.p.ndim:]
        # self._supp_flat, self._p_flat = self.supp.reshape((np.prod(self.p.shape), -1)), self.p.flatten()

    @property
    def mode(self):
        return self._mode

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov


#%% Deterministic RV, multivariate

# TODO: numeric/continuous only? change name?

# def _deterministic_check_parameters(val):
#     val = np.asarray(val)
#     if not np.issubdtype(val.dtype, np.number):
#         raise TypeError("Input 'val' must be of numeric type.")
#     return val

# def _deterministic_check_input(x, shape):
#     x, _ = _check_data_shape(x, shape)
#     # if not np.issubdtype(x.dtype, np.number):
#     #     raise TypeError("Input 'x' must be of numeric type.")
#     return x


class DeterministicRE(BaseRE):
    def __init__(self, val, seed=None):
        super().__init__(val, seed=seed)

    # Input properties
    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._check_inputs(val)
        self._update_attr()

    # Base method overwrites
    def _check_inputs(self, val):
        val = np.asarray(val)
        self._val = val

    def _update_attr(self):
        self._mode = self.val
        if np.issubdtype(self.val.dtype, np.number):
            self._mean = self.val
            self._cov = np.zeros(2 * self.val.shape)
        else:
            # warnings.warn("Method only supported for numeric 'val'.")
            self._mean = None
            self._cov = None

    def pmf(self, x):
        x, _ = _check_data_shape(x, self.val.shape)
        return np.where(np.all(x.reshape(-1, self.val.size) == self.val.flatten(), axis=-1).squeeze(), 1., 0.)

    def rvs(self, size=None):
        if size is None:
            return self.val
        else:
            return np.broadcast_to(self.val, (size,) + self.val.shape)


rng = np.random.default_rng()
a = np.arange(6).reshape(3, 2)
# a = ['a','b','c']
b = DeterministicRE(a[1], rng)
b.mode
b.mean
b.cov
b.pmf(a)
b.rvs(3)


#%% Discrete RV, multivariate (generalized)

# TODO: modify for non-scalar elements?

# TODO: use structured array to combine support and pmf?

def _discrete_check_parameters(supp, p):
    supp = np.asarray(supp)
    p = np.asarray(p)

    set_shape, data_shape = supp.shape[:p.ndim], supp.shape[p.ndim:]
    if set_shape != p.shape:
        raise ValueError("Leading shape values of 'supp' must equal the shape of 'pmf'.")

    supp_flat = supp.reshape((np.prod(set_shape), -1))
    if len(supp_flat) != len(np.unique(supp_flat, axis=0)):
        raise ValueError("Input 'supp' must have unique values")

    if np.min(p) < 0:
        raise ValueError("Each entry in 'pmf' must be greater than or equal "
                         "to zero.")
    if np.abs(p.sum() - 1.0) > 1e-9:
        raise ValueError("The input 'pmf' must lie within the normal "
                         "simplex. but pmf.sum() = %s." % p.sum())

    return supp, p


# def _discrete_multi_check_input(x, supp):
#     x, size = _check_data_shape(x, supp.shape)
#     if not np.isin(x, supp).all():
#         raise ValueError("Elements of input 'x' must be in the support set %s." % supp)     # TODO: flatten???
#
#     return x, size      # TODO: remove size?


class FiniteRE(BaseRE):
    def __init__(self, supp, p, seed=None):
        super().__init__(supp, p, seed=seed)

    # Input properties
    @property
    def supp(self):
        return self._supp

    @supp.setter
    def supp(self, supp):
        self._check_inputs(supp, self._p)
        self._update_attr()

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._check_inputs(self._supp, p)
        self._update_attr()

    # Base method overwrites
    def _check_inputs(self, supp, p):
        self._supp, self._p = _discrete_check_parameters(supp, p)

    def _update_attr(self):
        self._data_shape = self.supp.shape[self.p.ndim:]
        self._supp_flat, self._p_flat = self.supp.reshape((self.p.size, -1)), self.p.flatten()

        self._mode = self._supp_flat[np.argmax(self._p_flat)].reshape(self._data_shape)

        if np.issubdtype(self.supp.dtype, np.number):
            mean_flat = (self._p_flat[:, np.newaxis] * self._supp_flat).sum(axis=0)
            self._mean = mean_flat.reshape(self._data_shape)
            ctr_flat = self._supp_flat - mean_flat
            outer_flat = (ctr_flat.reshape(self.p.size, 1, -1) * ctr_flat[..., np.newaxis]).reshape(self.p.size, -1)
            self._cov = (self._p_flat[:, np.newaxis] * outer_flat).sum(axis=0).reshape(2 * self._data_shape)
        else:
            # warnings.warn("Method only supported for numeric 'supp'.")
            self._mean = None
            self._cov = None

    def pmf(self, x):
        x, _ = _check_data_shape(x, self._data_shape)
        if not np.isin(x, self.supp).all():
            raise ValueError("Elements of input 'x' must be in the support set %s." % self.supp)  # TODO: flatten???

        return self._p_flat[np.all(x.flatten() == self._supp_flat, axis=-1)]

    def rvs(self, size=None, random_state=None):
        random_state = self._get_random_state(random_state)
        i = random_state.choice(self.p.size, size, p=self._p_flat)
        if size is None:
            return self._supp_flat[i].reshape(self._data_shape)
        else:
            return self._supp_flat[i].reshape((size,) + self._data_shape)
        # return random_state.choice(self.supp.flatten(), size, p=self.p.flatten())



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
    x, _ = _check_data_shape(x, mean.shape)

    if np.min(x) < 0:
        raise ValueError("Each entry in 'x' must be greater than or equal "
                         "to zero.")

    if (np.abs(x.reshape(-1, mean.size).sum(-1) - 1.0) > 1e-9).any():
        raise ValueError("The sample values in 'x' must lie within the normal "
                         "simplex, but the sums = %s." % x.reshape(-1, mean.size).sum(-1).squeeze())

    if np.logical_and(x == 0, mean < 1 / alpha_0).any():
        raise ValueError("Each entry in 'x' must be greater than zero if its "
                         "mean is less than 1 / alpha_0.")

    return x


class DirichletRE(BaseRE):
    def __init__(self, alpha_0, mean, seed=None):
        super().__init__(alpha_0, mean, seed=seed)

    # Input properties
    @property
    def alpha_0(self):
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        # self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
        self._check_inputs(alpha_0, None)
        self._update_attr()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        # self._mean = _dirichlet_check_mean(mean)
        self._check_inputs(None, mean)
        self._update_attr()

    # Base method overwrites
    def _check_inputs(self, alpha_0, mean):
        # self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)
        # self._mean = _dirichlet_check_mean(mean)
        if alpha_0 is not None:
            self._alpha_0 = _dirichlet_check_alpha_0(alpha_0)       # TODO: better way, pack args?
        if mean is not None:
            self._mean = _dirichlet_check_mean(mean)

    def _update_attr(self):
        if np.min(self.mean) > 1 / self.alpha_0:
            self._mode = (self.mean - 1 / self.alpha_0) / (1 - self.mean.size / self.alpha_0)
        else:
            warnings.warn("Mode method currently supported for mean > 1/alpha_0 only")
            self._mode = None       # TODO: complete with general formula

        self._cov = (diag_gen(self.mean) - outer_gen(self.mean, self.mean)) / (self.alpha_0 + 1)

        self._beta_inv = gammaln(np.sum(self.alpha_0 * self.mean)) - np.sum(gammaln(self.alpha_0 * self.mean))

    def pdf(self, x):
        x = _dirichlet_multi_check_input(x, self.alpha_0, self.mean)

        log_pdf = self._beta_inv + np.sum(xlogy(self.alpha_0 * self.mean - 1, x).reshape(-1, self.mean.size), -1)
        return np.exp(log_pdf).squeeze()

    def rvs(self, size=None, random_state=None):
        random_state = self._get_random_state(random_state)
        if size is None:
            return random_state.dirichlet(self.alpha_0 * self.mean.flatten()).reshape(self.mean.shape)
        else:
            return random_state.dirichlet(self.alpha_0 * self.mean.flatten(), size).reshape((size,) + self.mean.shape)


a0 = 4
m = np.random.random((3, 2))
m = m / m.sum()
d = DirichletRE(a0, m)
d.mean
d.mode
d.cov
d.pdf(m)
d.rvs()
