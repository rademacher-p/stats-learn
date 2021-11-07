"""
Bayesian random elements.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats._multivariate import _PSD

from stats_learn.random import elements as rand_elements
from stats_learn import spaces
from stats_learn.util import RandomGeneratorMixin


# TODO: rename `model` attributes to `element`?


class Base(RandomGeneratorMixin, ABC):
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

    @abstractmethod
    def random_model(self, rng=None):
        raise NotImplementedError

    sample = rand_elements.Base.sample

    def _sample(self, n, rng):
        return self.random_model(rng)._sample(n, rng)

    def fit(self, d=None, warm_start=False):
        if d is None:
            d = np.array([])

        self._fit(d, warm_start)

    @abstractmethod
    def _fit(self, d, warm_start=False):
        raise NotImplementedError


class NormalLinear(Base):
    def __init__(self, prior_mean=np.zeros(1), prior_cov=np.eye(1), basis=None, cov=1., *, allow_singular=False,
                 rng=None):

        # Prior
        prior = rand_elements.Normal(prior_mean, prior_cov)
        super().__init__(prior, rng)
        if self.prior.ndim > 1:
            raise ValueError

        self.allow_singular = allow_singular

        _temp = np.array(cov).shape
        self._space = spaces.Euclidean(_temp[:int(len(_temp) / 2)])

        # Model
        if basis is None:
            self._basis = np.vstack((np.eye(self.prior.size),
                                     np.zeros((self.size - self.prior.size, self.prior.size))))
        else:
            self._basis = np.array(basis)

        self._set_cov(cov)
        self._set_prior_persistent_attr()

        # Learning
        self.posterior = rand_elements.Normal(self.prior_mean, self.prior_cov)
        self.posterior_model = rand_elements.NormalLinear(**self._prior_model_kwargs)

    # Methods
    def random_model(self, rng=None):
        rng = self._get_rng(rng)

        model_kwargs = {'basis': self.basis, 'cov': self.cov, 'rng': rng}
        rand_kwargs = {'weights': self.prior.sample(rng=rng)}

        return rand_elements.NormalLinear(**model_kwargs, **rand_kwargs)

    def _fit(self, d, warm_start=False):
        if not warm_start:  # reset learning attributes
            self.n = 0
            self._mean_data_temp = np.zeros(self.prior.shape)

        n = len(d)
        if n > 0:  # update data-dependent attributes
            self.n += n

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
                                               + self.n * self._basis_white.T @ self._basis_white)
            self.posterior_model.cov = self._make_posterior_model_cov(self.posterior.cov)

        self.posterior.mean = self.posterior.cov @ (self._cov_prior_inv @ self.prior_mean + self._mean_data_temp)
        self.posterior_model.weights = self.posterior.mean

    def _make_posterior_model_cov(self, cov_weight):
        return self.cov + (self.basis @ cov_weight @ self.basis.T).reshape(2 * self.shape)

    # Model parameters
    @property
    def basis(self):
        return self._basis

    @property
    def cov(self):
        return self._cov

    def _set_cov(self, val):
        self._cov = np.array(val)
        self._prec_U = _PSD(self._cov.reshape(2 * (self.size,)), allow_singular=self.allow_singular).U
        self._basis_white = np.dot(self.basis.T, self._prec_U).T

    @cov.setter
    def cov(self, val):
        self._set_cov(val)
        self._reset_posterior()

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


class Dirichlet(Base):
    def __init__(self, prior_mean, alpha_0, rng=None):
        super().__init__(prior=None, rng=rng)
        self._space = prior_mean.space

        _emp_dist = rand_elements.DataEmpirical([], [], space=self.space)
        self.posterior_model = rand_elements.Mixture([prior_mean, _emp_dist], [alpha_0, _emp_dist.n])

    def __repr__(self):
        return f"Dirichlet(alpha_0={self.alpha_0}, n={self.n}, prior_mean={self.prior_mean})"

    def __setattr__(self, name, value):
        if name.startswith('prior_mean.'):
            self.posterior_model.set_dist_attr(0, **{name.replace('prior_mean.', ''): value})
        else:
            super().__setattr__(name, value)

    @property
    def prior_mean(self):
        return self.posterior_model.dists[0]

    @prior_mean.setter
    def prior_mean(self, val):
        self.posterior_model.set_dist(0, val, self.alpha_0)

    @property
    def alpha_0(self):
        return self.posterior_model.weights[0]

    @alpha_0.setter
    def alpha_0(self, val):
        self.posterior_model.weights = [val, self.n]

    @property
    def emp_dist(self):
        return self.posterior_model.dists[1]

    @emp_dist.setter
    def emp_dist(self, val):
        self.posterior_model.set_dist(1, val, val.n)

    n = property(lambda self: self.emp_dist.n)

    def random_model(self, rng=None):
        raise NotImplementedError  # TODO: implement for finite in subclass?

    def _sample(self, n, rng):
        # Samples directly from the marginal Dirichlet-Empirical data distribution

        _out = np.empty((n, *self.shape), dtype=self.space.dtype)
        for i in range(n):
            if rng.random() <= self.alpha_0 / (self.alpha_0 + i):
                _out[i] = self.prior_mean.sample(rng=rng)  # sample from mean distribution
            else:
                _out[i] = rng.choice(_out[:i])

        return _out

    def _fit(self, d, warm_start=False):
        if warm_start:
            emp_dist = self.emp_dist
        else:
            emp_dist = rand_elements.DataEmpirical([], [], space=self.space)
        emp_dist.add_data(d)
        self.emp_dist = emp_dist
