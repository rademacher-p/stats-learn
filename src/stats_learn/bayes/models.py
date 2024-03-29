r"""
Bayesian random models.

Consist of jointly distributed random elements :math:`\mathrm{x}` and :math:`\mathrm{y}`
with prior sampling and posterior fitting.
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats._multivariate import _PSD

from stats_learn import random, spaces
from stats_learn.util import RandomGeneratorMixin, make_power_func

# TODO: Add deterministic DEP to effect a DP realization and sample!!


class Base(RandomGeneratorMixin, ABC):
    """
    Base class for Bayesian random models.

    Parameters
    ----------
    prior : stats_learn.random.elements.Base, optional
        Random element characterizing the prior distribution of the element parameters.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    _space: dict[str, spaces.Base | None]

    can_warm_start = False

    def __init__(self, prior=None, rng=None):
        super().__init__(rng)

        self._space = {"x": None, "y": None}

        self.prior = prior
        self.posterior = None
        self.posterior_model = None

    @property
    def space(self):
        """
        The domain.

        Returns
        -------
        dict of spaces.Base

        """
        return self._space

    @property
    def shape(self):
        """
        Shape of the random models.

        Returns
        -------
        dict of tuple

        """
        return {key: space.shape for key, space in self._space.items()}

    @property
    def size(self):
        """
        Size of random models.

        Returns
        -------
        dict of int

        """
        return {key: space.size for key, space in self._space.items()}

    @property
    def ndim(self):
        """
        Dimensionality of random models.

        Returns
        -------
        dict of int

        """
        return {key: space.ndim for key, space in self._space.items()}

    @property
    def dtype(self):
        """
        Data type of random models.

        Returns
        -------
        dict of np.dtype

        """
        return {key: space.dtype for key, space in self._space.items()}

    def random_model(self, rng=None):
        """Generate a random element with a randomly selected parameterization."""
        raise NotImplementedError

    def sample(self, size=None, rng=None):
        return random.elements.Base.sample(self, size, rng)

    def _sample(self, size, rng):
        model = self.random_model(rng)
        return model.sample(size)

    def fit(self, d=None, warm_start=False):
        """
        Refine the posterior using observations.

        Parameters
        ----------
        d : np.ndarray, optional
            The observations.
        warm_start : bool, optional
            If `False`, `reset` is invoked to restore unfit state.

        """
        if not warm_start:
            self.reset()
        elif not self.can_warm_start:
            raise ValueError("Bayes model does not support warm start fitting.")

        if d is not None and len(d) > 0:
            self._fit(d)

    @abstractmethod
    def _fit(self, d):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Restore unfit prior state."""
        raise NotImplementedError

    def _format_params(self, key, value=None):
        return key, value


class NormalLinear(Base):
    r"""
    Random model characterized by a Normal conditional distribution.

    Mean is linear in basis weights. User defines basis functions and parameterizes
    Normal prior distribution for weights.

    Parameters
    ----------
    prior_mean : array_like, optional
        Mean of Normal prior random variable.
    prior_cov : array_like, optional
        Covariance of Normal prior random variable.
    basis_y_x : Collection of callable, optional
        Basis functions. Defaults to polynomial functions.
    cov_y_x : float or numpy.ndarray, optional
        Conditional covariance of Normal distributions.
    model_x : stats_learn.random.elements.Base, optional
        Random variable for the marginal distribution of :math:`\mathrm{x}`.
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
        basis_y_x=None,
        cov_y_x=1.0,
        model_x=None,
        *,
        allow_singular=True,
        rng=None,
    ):
        self.allow_singular = allow_singular

        # Prior
        prior = random.elements.Normal(
            prior_mean, prior_cov, allow_singular=self.allow_singular
        )
        super().__init__(prior, rng)
        if self.prior.ndim > 1:
            raise ValueError

        # Model
        if model_x is None:
            model_x = random.elements.Normal()
        self._model_x = model_x
        self._space["x"] = model_x.space

        _temp = np.array(cov_y_x).shape
        self._space["y"] = spaces.Euclidean(_temp[: int(len(_temp) / 2)])

        self._set_cov_y_x(cov_y_x)

        if basis_y_x is None:
            basis_y_x = tuple(map(make_power_func, range(len(self.prior_mean))))

        self._basis_y_x = basis_y_x
        # self._basis_y_x = tuple(
        #     vectorize_func(f, shape=self.shape["x"]) for f in basis_y_x
        # )

        self._set_prior_persistent_attr()

        # Learning
        self._cov_data_inv = np.zeros(2 * self.prior.shape)
        self._mean_data_temp = np.zeros(self.prior.shape)

        self.posterior = random.elements.Normal(
            self.prior_mean, self.prior_cov, allow_singular=self.allow_singular
        )
        self.posterior_model = random.models.NormalLinear(**self._prior_model_kwargs)

    # Methods
    def random_model(self, rng=None):
        """Generate a random element with a randomly selected parameterization."""
        rng = self._get_rng(rng)

        model_kwargs = {
            "basis_y_x": self.basis_y_x,
            "cov_y_x": self.cov_y_x,
            "model_x": self.model_x,
            "rng": rng,
        }
        rand_kwargs = {"weights": self.prior.sample(rng=rng)}

        return random.models.NormalLinear(**model_kwargs, **rand_kwargs)

    def reset(self):
        """Restore unfit prior state."""
        self._cov_data_inv = np.zeros(2 * self.prior.shape)
        self._mean_data_temp = np.zeros(self.prior.shape)
        self._reset_posterior()

    def _fit(self, d):
        psi = np.array(
            [np.array([func(x_i) for func in self.basis_y_x]) for x_i in d["x"]]
        ).reshape((len(d), self.prior.size, self.size["y"]))
        psi_white = np.dot(psi, self._prec_U_y_x)
        self._cov_data_inv += sum(psi_i @ psi_i.T for psi_i in psi_white)

        y_white = np.dot(d["y"].reshape(len(d), self.size["y"]), self._prec_U_y_x)
        self._mean_data_temp += sum(
            psi_i @ y_i for psi_i, y_i in zip(psi_white, y_white)
        )

        self._update_posterior()

    def _reset_posterior(self):
        self.posterior.mean = self.prior_mean
        self.posterior.cov = self.prior_cov

        kwargs = self._prior_model_kwargs.copy()
        del kwargs["basis_y_x"]
        for key, value in kwargs.items():
            setattr(self.posterior_model, key, value)

    @property
    def _prior_model_kwargs(self):
        return {
            "weights": self.prior_mean,
            "basis_y_x": self.basis_y_x,
            "cov_y_x": self._prior_model_cov,
            "model_x": self.model_x,
        }

    def _update_posterior(self, mean_only=False):
        if not mean_only:
            self.posterior.cov = np.linalg.inv(self._cov_prior_inv + self._cov_data_inv)
            self.posterior_model.cov_y_x_ = self._make_posterior_model_cov(
                self.posterior.cov
            )

        self.posterior.mean = self.posterior.cov @ (
            self._cov_prior_inv @ self.prior_mean + self._mean_data_temp
        )
        self.posterior_model.weights = self.posterior.mean

    def _make_posterior_model_cov(self, cov_weight):
        def cov_y_x(x):
            psi_x = np.array([func(x) for func in self.basis_y_x]).reshape(
                self.prior.size, self.size["y"]
            )
            cov_add = (psi_x.T @ cov_weight @ psi_x).reshape(2 * self.shape["y"])
            return self.cov_y_x + cov_add

        return cov_y_x

    # Model parameters
    @property
    def model_x(self):
        r"""Random variable for the marginal distribution of :math:`\mathrm{x}`."""
        return self._model_x

    @model_x.setter
    def model_x(self, value):
        self._model_x = value
        self._reset_posterior()

    @property
    def basis_y_x(self):
        """Basis functions."""
        return self._basis_y_x

    @property
    def cov_y_x(self):
        r"""Covariance tensor characterizing fixed variance of :math:`\mathrm{y}`."""
        return self._cov_y_x

    def _set_cov_y_x(self, value):
        self._cov_y_x = np.array(value)
        self._prec_U_y_x = _PSD(
            self._cov_y_x.reshape(2 * (self.size["y"],)),
            allow_singular=self.allow_singular,
        ).U

    @cov_y_x.setter  # type: ignore
    def cov_y_x(self, value):
        self._set_cov_y_x(value)
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

    def _format_params(self, key, value=None):
        """Format attributes as strings for TeX."""
        str_theta = r"\theta"
        str_one = ""
        str_eye = ""
        if plt.rcParams["text.usetex"]:
            if "upgreek" in plt.rcParams["text.latex.preamble"]:
                str_theta = r"\uptheta"
            if "bm" in plt.rcParams["text.latex.preamble"]:
                str_one = r"\bm{1}"
                str_eye = r"\bm{I}"

        if key == "prior_mean":
            key = rf"\mu_{str_theta}"
            if value is not None:
                val_np = np.array(value)
                value = f"{value:.3f}"
                if self.prior.shape != () and val_np.shape == ():
                    value += str_one

        elif key == "prior_cov":
            key = rf"\Sigma_{str_theta}"
            if value is not None:
                val_np = np.array(value)
                value = f"{value:.3f}"
                if self.prior.shape != () and val_np.shape == ():
                    value += str_eye

        return key, value


class Dirichlet(Base):
    """
    Generic random model whose joint distribution is a Dirichlet process.

    Parameters
    ----------
    prior_mean : stats_learn.random.models.Base
        Random model characterizing the mean of the Dirichlet process.
    alpha_0 : float
        Dirichlet localization (i.e. concentration) parameter.
    rng : np.random.Generator or int, optional
        Random number generator seed or object.

    """

    can_warm_start = True

    def __init__(self, prior_mean, alpha_0, rng=None):
        super().__init__(prior=None, rng=rng)

        self._space = prior_mean.space

        _emp_dist = random.models.DataEmpirical([], [], space=self.space)
        self.posterior_model = random.models.Mixture(
            [prior_mean, _emp_dist], [alpha_0, _emp_dist.n]
        )

    def __repr__(self):
        _strs = ["alpha_0", "n", "prior_mean"]
        param_strs = (f"{s}={getattr(self, s)}" for s in _strs)
        return f"Dirichlet({', '.join(param_strs)})"

    def __setattr__(self, name, value):
        if name.startswith("prior_mean."):
            _kwargs = {name.replace("prior_mean.", ""): value}
            self.posterior_model.set_dist_attr(0, **_kwargs)
        else:
            super().__setattr__(name, value)

    @property
    def prior_mean(self):
        """Random model characterizing the mean of the Dirichlet process."""
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
        """Generate a random model with a randomly selected parameterization."""
        raise NotImplementedError  # TODO: implement for finite in subclass?

    def _sample(self, n, rng):
        # Samples directly from the marginal Dirichlet-Empirical data distribution

        _data = [
            tuple(np.empty(self.shape[c], self.dtype[c]) for c in "xy")
            for _ in range(n)
        ]
        _out = np.array(_data, dtype=[(c, self.dtype[c], self.shape[c]) for c in "xy"])
        for i in range(n):
            if rng.random() <= (1 + i / self.alpha_0) ** -1:
                # sample from mean distribution
                _out[i] = self.prior_mean.sample(rng=rng)
            else:
                _out[i] = rng.choice(_out[:i])

        return _out

    def reset(self):
        """Restore unfit prior state."""
        self.emp_dist = random.models.DataEmpirical([], [], space=self.space)

    def _fit(self, d):
        emp_dist = self.emp_dist
        emp_dist.add_data(d)
        self.emp_dist = emp_dist  # triggers setter

    @staticmethod
    def _format_params(key, value=None):
        """Format attributes as strings for TeX."""
        if key == "alpha_0":
            key = r"\alpha_0"
        return key, value
