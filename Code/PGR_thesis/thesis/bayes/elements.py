"""
Bayesian random elements.
"""

import math

import numpy as np
from scipy.stats._multivariate import _PSD

from thesis.random.elements import Normal, Base as BaseRE, NormalLinear as NormalLinearRE, GenericEmpirical
from thesis.util.generic import RandomGeneratorMixin
from thesis.util import spaces

np.set_printoptions(precision=2)


#%% Priors

class Base(RandomGeneratorMixin):
    # param_names = ()

    def __init__(self, prior=None, rng=None):
        super().__init__(rng)

        self._space = None

        self.prior = prior
        self.posterior = None
        self.posterior_model = None

    space = property(lambda self: self._space)

    shape = property(lambda self: self._space.shape)
    size = property(lambda self: self._space.size)
    ndim = property(lambda self: self._space.ndim)

    def random_model(self, rng=None):
        raise NotImplementedError

    rvs = BaseRE.rvs

    def _rvs(self, n, rng):
        return self.random_model(rng)._rvs(n, rng)

    def fit(self, d=None, warm_start=False):
        if d is None:
            d = np.array([])

        self._fit(d, warm_start)

    def _fit(self, d, warm_start=False):
        raise NotImplementedError


class NormalLinear(Base):
    def __init__(self, prior_mean=np.zeros(1), prior_cov=np.eye(1), basis=None, cov=1., rng=None):

        # Prior
        prior = Normal(prior_mean, prior_cov)
        super().__init__(prior, rng)
        if self.prior.ndim > 1:
            raise ValueError

        # Model
        self._set_cov(cov)
        self._set_basis(basis)

        self._set_prior_persistent_attr()
        self._set_basis_white()

        # Learning
        self.posterior = Normal(self.prior_mean, self.prior_cov)
        self.posterior_model = NormalLinearRE(**self._prior_model_kwargs)

    # Methods
    def random_model(self, rng=None):
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
        self.posterior.mean = self.prior_mean
        self.posterior.cov = self.prior_cov

        for key, val in self._prior_model_kwargs.items():
            setattr(self.posterior_model, key, val)

    @property
    def _prior_model_kwargs(self):
        return {'weights': self.prior_mean, 'basis': self.basis, 'cov': self._prior_model_cov}

    def _update_posterior(self, mean_only=False):
        if not mean_only:
            self.posterior.cov = np.linalg.inv(self._cov_prior_inv
                                               + self._n_total * self._basis_white.T @ self._basis_white)
            self.posterior_model.cov = self._make_posterior_model_cov(self.posterior.cov)

        self.posterior.mean = self.posterior.cov @ (self._cov_prior_inv @ self.prior_mean + self._mean_data_temp)
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
        self._space = spaces.Euclidean(_temp[:int(len(_temp) / 2)])

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
    def prior_mean(self):
        return self.prior.mean

    @prior_mean.setter
    def prior_mean(self, val):
        self.prior.mean = val
        self._update_posterior(mean_only=True)

    @property
    def prior_cov(self):
        return self.prior.cov

    @prior_cov.setter
    def prior_cov(self, val):
        self.prior.cov = val
        self._set_prior_persistent_attr()
        self._update_posterior()

    def _set_prior_persistent_attr(self):
        self._cov_prior_inv = np.linalg.inv(self.prior_cov)
        self._prior_model_cov = self._make_posterior_model_cov(self.prior_cov)


# basis = [[1, 0], [0, 1], [1, 1]]
# a = NormalLinear(prior_mean=np.ones(2), prior_cov=10*np.eye(2), basis=basis, cov=np.eye(3), rng=None)
# r = a.random_model()
# d = r.rvs(10)
# a.fit(d)
# print(a.prior.mean)
# print(a.posterior.mean)
# print(r.weights)


class Dirichlet(Base):
    def __init__(self, alpha_0, prior_mean, rng=None):
        super().__init__(prior=None, rng=rng)
        self.alpha_0 = alpha_0
        self.prior_mean = prior_mean

        # self._space = spaces.Euclidean(self.prior_mean.shape)
        self._space = self.prior_mean.space

        # Learning
        # self.posterior = None

        self.emp_dist = None
        self.posterior_model = None     # TODO: mixture dist

    def random_model(self, rng=None):
        raise NotImplementedError       # TODO: implement for finite in subclass?

    def _rvs(self, n, rng):
        # Samples directly from the marginal data distribution
        _out = np.empty((n, *self.shape))
        for i in range(n):
            if rng.random() <= self.alpha_0 / (self.alpha_0 + i):
                _out[i] = self.prior_mean.rvs(rng=rng)     # sample from mean distribution
            else:
                _out[i] = rng.choice(_out[:i])

        return _out

    def _fit(self, d, warm_start=False):
        self.emp_dist = GenericEmpirical(d)        # TODO: in-place?
