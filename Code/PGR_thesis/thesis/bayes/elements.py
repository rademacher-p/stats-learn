"""
Bayesian random elements.
"""

import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats._multivariate import _PSD

from thesis.random import elements as rand_elements
from thesis.util.base import RandomGeneratorMixin
from thesis.util import spaces

np.set_printoptions(precision=2)

# TODO: rename `model` attributes to `element`?


#%% Priors

class Base(RandomGeneratorMixin):
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

    rvs = rand_elements.Base.rvs

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
        prior = rand_elements.Normal(prior_mean, prior_cov)
        super().__init__(prior, rng)
        if self.prior.ndim > 1:
            raise ValueError

        # Model
        self._set_cov(cov)
        self._set_basis(basis)

        self._set_prior_persistent_attr()
        self._set_basis_white()

        # Learning
        self.posterior = rand_elements.Normal(self.prior_mean, self.prior_cov)
        self.posterior_model = rand_elements.NormalLinear(**self._prior_model_kwargs)

    # Methods
    def random_model(self, rng=None):
        rng = self._get_rng(rng)

        model_kwargs = {'basis': self.basis, 'cov': self.cov, 'rng': rng}
        rand_kwargs = {'weights': self.prior.rvs(rng=rng)}

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

    def _set_basis(self, val):
        if val is None:
            # self._basis = np.ones((self.size, self.prior.size))
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

        self._space = self.prior_mean.space

        # Learning
        self.posterior = None

        _emp_dist = rand_elements.DataEmpirical([], [], space=self.space)
        self.posterior_model = rand_elements.Mixture([self.prior_mean, _emp_dist], [self.alpha_0, 0])

    # def __getattribute__(self, name):
    #     try:
    #         return getattr(self.prior_mean, name)
    #     except AttributeError:
    #         return super().__getattribute__(name)

    # def __setattr__(self, name, value):
    #     try:
    #         self.posterior_model.set_dist_attr(0, **{name: value})      # prior attributes take precedence
    #     except AttributeError:
    #         super().__setattr__(name, value)

    def __setattr__(self, name, value):     # TODO: better way?
        if name == 'alpha_0':
            super().__setattr__(name, value)
        else:
            try:
                self.posterior_model.set_dist_attr(0, **{name: value})
            except AttributeError:
                super().__setattr__(name, value)

    @property
    def emp_dist(self):
        return self.posterior_model.dists[1]

    @emp_dist.setter
    def emp_dist(self, val):
        self.posterior_model.set_dist(1, val, val.n)

    def random_model(self, rng=None):
        raise NotImplementedError       # TODO: implement for finite in subclass?

    def _rvs(self, n, rng):
        # Samples directly from the marginal Dirichlet-Empirical data distribution

        _out = np.empty((n, *self.shape), dtype=self.space.dtype)
        for i in range(n):
            if rng.random() <= self.alpha_0 / (self.alpha_0 + i):
                _out[i] = self.prior_mean.rvs(rng=rng)     # sample from mean distribution
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

        # if not warm_start:
        #     self.emp_dist = rand_elements.DataEmpirical([], [], space=self.space)
        # self.emp_dist.add_data(d)
        # self.posterior_model._update_attr()


if __name__ == '__main__':
    # alpha = rand_elements.Beta(5, 25)
    # theta = rand_elements.Beta(25, 5)

    alpha = rand_elements.Finite(['a', 'b'], [.2, .8])
    theta = rand_elements.Finite(['a', 'b'], [.8, .2])

    print(f"Mode = {theta.mode}")
    # print(f"Mean = {theta.mean}")
    theta.plot_pf()
    plt.title("True")

    a = Dirichlet(alpha_0=10, prior_mean=alpha)

    # a.rvs(5)
    print(f"Mode = {a.posterior_model.mode}")
    # print(f"Mean = {a.posterior_model.mean}")
    a.posterior_model.plot_pf()
    plt.title("Prior")

    a.fit(theta.rvs(100))
    # a.rvs(10)
    print(f"Mode = {a.posterior_model.mode}")
    # print(f"Mean = {a.posterior_model.mean}")
    a.posterior_model.plot_pf()
    plt.title("Posterior")
    pass
