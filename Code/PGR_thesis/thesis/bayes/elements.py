"""
Bayesian random elements.
"""

import math

import numpy as np
from scipy.stats._multivariate import _PSD

from thesis.random.elements import Normal, Base as BaseRE, NormalLinear as NormalLinearRE
from thesis.util.generic import RandomGeneratorMixin

#%% Priors


class Base(RandomGeneratorMixin):
    # param_names = ()

    def __init__(self, prior=None, rng=None):
        super().__init__(rng)

        self._shape = None

        self.prior = prior
        self.posterior = None

    shape = property(lambda self: self._shape)
    size = property(lambda self: math.prod(self._shape))
    ndim = property(lambda self: len(self._shape))

    def random_element(self, rng=None):
        raise NotImplementedError

    rvs = BaseRE.rvs

    def _rvs(self, size, rng):
        model = self.random_element(rng)
        return model._rvs(size)

    def fit(self, d=None, warm_start=False):
        if d is None:
            d = np.array([])

        self._fit(d, warm_start)

    def _fit(self, d, warm_start=False):
        raise NotImplementedError


class NormalLinear(Base):
    def __init__(self, mean_prior=np.zeros(1), cov_prior=np.eye(1), basis=None, cov=1., rng=None):

        # Prior
        prior = Normal(mean_prior, cov_prior)
        super().__init__(prior, rng)
        if self.prior.ndim > 1:
            raise ValueError

        # Model
        self._set_cov(cov)
        self._set_basis(basis)

        self._set_prior_persistent_attr()
        self._set_basis_white()

        # Learning
        self.posterior = Normal(self.mean_prior, self.cov_prior)
        self.posterior_model = NormalLinearRE(**self._prior_model_kwargs)

    # Methods
    def random_element(self, rng=None):
        rng = self._get_rng(rng)

        model_kwargs = {'basis': self.basis, 'cov': self.cov, 'rng': rng}
        rand_kwargs = {'weights': self.prior.rvs(rng=rng)}

        return NormalLinearRE(**model_kwargs, **rand_kwargs)

    def _fit(self, d, warm_start=False):
        if not warm_start:  # reset learning attributes
            self._n_total = 0
            self._mean_data_temp = np.zeros(self.prior.shape)

        n = len(d)
        if n > 0:  # update data-dependent attributes
            self._n_total += n

            y_white = np.dot(d.reshape(n, self.size), self._prec_U)
            self._mean_data_temp += sum(self._basis_white.T @ y_i for y_i in y_white)

        self._update_posterior()

    def _reset_posterior(self):
        self.posterior.mean = self.mean_prior
        self.posterior.cov = self.cov_prior

        for key, val in self._prior_model_kwargs.items():
            setattr(self.posterior_model, key, val)

    @property
    def _prior_model_kwargs(self):
        return {'weights': self.mean_prior, 'basis': self.basis, 'cov': self._prior_model_cov}

    def _update_posterior(self, mean_only=False):
        if not mean_only:
            self.posterior.cov = np.linalg.inv(self._cov_prior_inv
                                               + self._n_total * self._basis_white.T @ self._basis_white)
            self.posterior_model.cov = self._make_posterior_model_cov(self.posterior.cov)

        self.posterior.mean = self.posterior.cov @ (self._cov_prior_inv @ self.mean_prior + self._mean_data_temp)
        self.posterior_model.weights = self.posterior.mean

    def _make_posterior_model_cov(self, cov_weight):
        return self.cov + (self.basis @ cov_weight @ self.basis.T).reshape(2 * self.shape)

    # Model parameters
    @property
    def basis(self):
        return self._basis

    def _set_basis(self, val):
        if val is None:
            self._basis = np.ones((self.size, self.prior.size))
            self._basis = np.vstack((np.eye(self.prior.size), np.zeros((self.size - self.prior.size, self.prior.size))))
        else:
            self._basis = np.array(val)

    @basis.setter
    def basis(self, val):
        self._set_basis(val)
        self._set_basis_white()
        self._reset_posterior()

    @property
    def cov(self):
        return self._cov

    def _set_cov(self, val):
        self._cov = np.array(val)

        _temp = self._cov.shape
        self._shape = _temp[:int(len(_temp) / 2)]

        self._prec_U = _PSD(self._cov.reshape(2 * (self.size,)), allow_singular=False).U

    @cov.setter
    def cov(self, val):
        self._set_cov(val)
        self._set_basis_white()
        self._reset_posterior()

    def _set_basis_white(self):
        self._basis_white = np.dot(self.basis.T, self._prec_U).T

    # Prior parameters
    @property
    def mean_prior(self):
        return self.prior.mean

    @mean_prior.setter
    def mean_prior(self, val):
        self.prior.mean = val
        self._update_posterior(mean_only=True)

    @property
    def cov_prior(self):
        return self.prior.cov

    @cov_prior.setter
    def cov_prior(self, val):
        self.prior.cov = val
        self._set_prior_persistent_attr()
        self._update_posterior()

    def _set_prior_persistent_attr(self):
        self._cov_prior_inv = np.linalg.inv(self.cov_prior)
        self._prior_model_cov = self._make_posterior_model_cov(self.cov_prior)


# basis = [[1, 0], [0, 1], [1, 1]]
# a = NormalLinear(mean_prior=np.ones(2), cov_prior=10*np.eye(2), basis=basis, cov=np.eye(3), rng=None)
# r = a.random_element()
# d = r.rvs(10)
# a.fit(d)
# print(a.prior.mean)
# print(a.posterior.mean)
# print(r.weights)
