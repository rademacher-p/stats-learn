"""Bayesian random elements with prior sampling and posterior fitting."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats._multivariate import _PSD

from stats_learn import random, spaces
from stats_learn.util import RandomGeneratorMixin

# TODO: rename `model` attributes to `element`? Find common term for both?


class Base(RandomGeneratorMixin, ABC):
    """
    Base class for Bayesian random elements.

    Parameters
    ----------
    prior : stats_learn.random.elements.Base, optional
        Random element characterizing the prior distribution of the element parameters.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    Attributes
    ----------
    prior : stats_learn.random.elements.Base, optional
        Random element characterizing the prior distribution of the element parameters.

    """

    can_warm_start = False

    def __init__(self, prior=None, rng=None):
        super().__init__(rng)

        self._space = None

        self.prior = prior
        self.posterior = None  # the posterior distribution of the element parameters.
        self.posterior_model = None  # the posterior element given the observations.

    @property
    def space(self):
        """
        The domain.

        Returns
        -------
        spaces.Base

        """
        return self._space

    @property
    def shape(self):
        """
        Shape of the random elements.

        Returns
        -------
        tuple

        """
        return self.space.shape

    @property
    def size(self):
        """
        Size of random elements.

        Returns
        -------
        int

        """
        return self.space.size

    @property
    def ndim(self):
        """
        Dimensionality of random elements.

        Returns
        -------
        int

        """
        return self.space.ndim

    @property
    def dtype(self):
        """
        Data type of random elements.

        Returns
        -------
        np.dtype

        """
        return self.space.dtype

    @abstractmethod
    def random_model(self, rng=None):
        """Generate a random element with a randomly selected parameterization."""
        raise NotImplementedError

    sample = random.elements.Base.sample

    def _sample(self, n, rng):
        return self.random_model(rng)._sample(n, rng)

    def fit(self, d=None, warm_start=False):
        """
        Refine the posterior using observations.

        Parameters
        ----------
        d : array_like, optional
            The observations.
        warm_start : bool, optional
            If `False`, `reset` is invoked to restore unfit state.

        """
        if not warm_start:
            self.reset()
        elif not self.can_warm_start:
            raise ValueError("Bayes element does not support warm start fitting.")

        if d is not None and len(d) > 0:
            self._fit(d)

    @abstractmethod
    def _fit(self, d):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Restore unfit prior state."""
        raise NotImplementedError


class NormalLinear(Base):
    """
    Normal random variable with a mean linear in basis weights.

    User defines basis tensors and parameterizes Normal prior distribution for weights.

    Parameters
    ----------
    prior_mean : array_like, optional
        Mean of Normal prior random variable.
    prior_cov : array_like, optional
        Covariance of Normal prior random variable.
    basis : array_like, optional
        Basis tensors, such that `mean = basis @ weights`. Defaults to Euclidean basis.
    cov : float or numpy.ndarray, optional
        Covariance tensor.
    allow_singular : bool, optional
        Whether to allow a singular prior covariance matrix.
    rng : np.random.Generator or int, optional
        Random number generator seed or object.

    """

    prior: random.elements.Normal
    can_warm_start = True

    def __init__(
        self,
        prior_mean=(0.0,),
        prior_cov=((1.0,),),
        basis=None,
        cov=1.0,
        *,
        allow_singular=True,
        rng=None,
    ):
        # Prior
        prior = random.elements.Normal(prior_mean, prior_cov)
        super().__init__(prior, rng)
        if self.prior.ndim > 1:
            raise NotImplementedError("Only 1-dimensional priors are supported.")

        self.allow_singular = allow_singular

        _temp = np.array(cov).shape
        self._space = spaces.Euclidean(_temp[: int(len(_temp) / 2)])

        # Model
        if basis is None:
            self._basis = np.concatenate(
                (
                    np.eye(self.prior.size),
                    np.zeros((self.size - self.prior.size, self.prior.size)),
                )
            )
        else:
            self._basis = np.array(basis)

        self._set_cov(cov)
        self._set_prior_persistent_attr()

        # Learning
        self._n = 0
        self._mean_data_temp = np.zeros(self.prior.shape)

        self.posterior = random.elements.Normal(self.prior_mean, self.prior_cov)
        self.posterior_model = random.elements.NormalLinear(**self._prior_model_kwargs)

    # Methods
    def random_model(self, rng=None):
        """Generate a random element with a randomly selected parameterization."""
        rng = self._get_rng(rng)

        model_kwargs = {"basis": self.basis, "cov": self.cov, "rng": rng}
        rand_kwargs = {"weights": self.prior.sample(rng=rng)}

        return random.elements.NormalLinear(**model_kwargs, **rand_kwargs)

    def reset(self):
        """Restore unfit prior state."""
        self._n = 0
        self._mean_data_temp = np.zeros(self.prior.shape)
        self._reset_posterior()

    def _fit(self, d):
        self._n += len(d)

        y_white = np.dot(d.reshape(len(d), self.size), self._prec_U)
        self._mean_data_temp += sum(self._basis_white.T @ y_i for y_i in y_white)

        self._update_posterior()

    def _reset_posterior(self):
        self.posterior.mean = self.prior_mean
        self.posterior.cov = self.prior_cov

        kwargs = self._prior_model_kwargs.copy()
        del kwargs["basis"]
        for key, value in kwargs.items():
            setattr(self.posterior_model, key, value)

    @property
    def _prior_model_kwargs(self):
        return {
            "weights": self.prior_mean,
            "basis": self.basis,
            "cov": self._prior_model_cov,
        }

    def _update_posterior(self, mean_only=False):
        if not mean_only:
            self.posterior.cov = np.linalg.inv(
                self._cov_prior_inv + self._n * self._basis_white.T @ self._basis_white
            )
            self.posterior_model.cov = self._make_posterior_model_cov(
                self.posterior.cov
            )

        self.posterior.mean = self.posterior.cov @ (
            self._cov_prior_inv @ self.prior_mean + self._mean_data_temp
        )
        self.posterior_model.weights = self.posterior.mean

    def _make_posterior_model_cov(self, cov_weight):
        cov_add = (self.basis @ cov_weight @ self.basis.T).reshape(2 * self.shape)
        return self.cov + cov_add

    # Model parameters
    @property
    def basis(self):
        """Basis tensors, such that `mean = basis @ weights`."""
        return self._basis

    @property
    def cov(self):
        return self._cov

    def _set_cov(self, value):
        self._cov = np.array(value)
        self._prec_U = _PSD(
            self._cov.reshape(2 * (self.size,)), allow_singular=self.allow_singular
        ).U
        self._basis_white = np.dot(self.basis.T, self._prec_U).T

    @cov.setter
    def cov(self, value):
        self._set_cov(value)
        self._reset_posterior()

    # Prior parameters
    @property
    def prior_mean(self):
        """Access the mean of the `prior`."""
        return self.prior.mean

    @prior_mean.setter
    def prior_mean(self, value):
        self.prior.mean = value
        self._update_posterior(mean_only=True)

    @property
    def prior_cov(self):
        """Access the covariance of the `prior`."""
        return self.prior.cov

    @prior_cov.setter
    def prior_cov(self, value):
        self.prior.cov = value
        self._set_prior_persistent_attr()
        self._update_posterior()

    def _set_prior_persistent_attr(self):
        self._cov_prior_inv = np.linalg.inv(self.prior_cov)
        self._prior_model_cov = self._make_posterior_model_cov(self.prior_cov)


class Dirichlet(Base):
    """
    Generic random element whose distribution is characterized by a Dirichlet process.

    Parameters
    ----------
    prior_mean : stats_learn.random.elements.Base
        Random element characterizing the mean of the Dirichlet process.
    alpha_0 : float
        Dirichlet localization (i.e. concentration) parameter.
    rng : np.random.Generator or int, optional
        Random number generator seed or object.

    """

    can_warm_start = True

    def __init__(self, prior_mean, alpha_0, rng=None):
        super().__init__(prior=None, rng=rng)
        self._space = prior_mean.space

        _emp_dist = random.elements.DataEmpirical([], [], space=self.space)
        self.posterior_model = random.elements.Mixture(
            [prior_mean, _emp_dist], [alpha_0, _emp_dist.n]
        )

    def __repr__(self):
        return f"Dirichlet(alpha_0={self.alpha_0}, n={self.n}, prior_mean={self.prior_mean})"

    def __setattr__(self, name, value):
        if name.startswith("prior_mean."):
            _kwargs = {name.replace("prior_mean.", ""): value}
            self.posterior_model.set_dist_attr(0, **_kwargs)
        else:
            super().__setattr__(name, value)

    @property
    def prior_mean(self):
        """Random element characterizing the mean of the Dirichlet process."""
        return self.posterior_model.dists[0]

    @prior_mean.setter
    def prior_mean(self, value):
        self.posterior_model.set_dist(0, value, self.alpha_0)

    @property
    def alpha_0(self):
        """Dirichlet localization (i.e. concentration) parameter."""
        return self.posterior_model.weights[0]

    @alpha_0.setter
    def alpha_0(self, value):
        self.posterior_model.weights = [value, self.n]

    @property
    def emp_dist(self):
        """The empirical distribution formed by observations."""
        return self.posterior_model.dists[1]

    @emp_dist.setter
    def emp_dist(self, value):
        self.posterior_model.set_dist(1, value, value.n)

    n = property(lambda self: self.emp_dist.n)

    def random_model(self, rng=None):
        """Generate a random element with a randomly selected parameterization."""
        raise NotImplementedError  # TODO: implement for finite in subclass?

    def _sample(self, n, rng):
        # Samples directly from the marginal Dirichlet-Empirical data distribution

        _out = np.empty((n, *self.shape), dtype=self.space.dtype)
        for i in range(n):
            if rng.random() <= self.alpha_0 / (self.alpha_0 + i):
                # sample from mean distribution
                _out[i] = self.prior_mean.sample(rng=rng)
            else:
                _out[i] = rng.choice(_out[:i])

        return _out

    def reset(self):
        """Restore unfit prior state."""
        self.emp_dist = random.elements.DataEmpirical([], [], space=self.space)

    def _fit(self, d):
        emp_dist = self.emp_dist
        emp_dist.add_data(d)
        self.emp_dist = emp_dist  # triggers setter
