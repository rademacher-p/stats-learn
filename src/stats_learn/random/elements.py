"""Random elements with sampling, statistic generation, and visualization tools."""

import math
from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np
from scipy.special import betaln, gammaln, xlog1py, xlogy
from scipy.stats._multivariate import _PSD

from stats_learn import spaces
from stats_learn.util import (
    RandomGeneratorMixin,
    check_data_shape,
    check_valid_pmf,
    vectorize_func,
)


class Base(RandomGeneratorMixin, ABC):
    """
    Base class for random elements.

    Parameters
    ----------
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, rng=None):
        super().__init__(rng)

        self._space = None  # TODO: arg?
        self._mode = None  # TODO: make getter do numerical approx if None!?

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

    @property  # TODO: `cached_property`?
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

    @property
    def mode(self):
        """The most probable domain value."""
        return self._mode

    def prob(self, x):
        """
        Probability function.

        Parameters
        ----------
        x : array_like
            Random element domain values.

        Returns
        -------
        np.ndarray
            Probability mass if `self.space` is a `spaces.Discrete` subclass;
            probability density if `spaces.Continuous` subclass.

        """
        # TODO: perform input checks using `space.__contains__`?

        # if x is None:
        #     x = self.space.x_plt  # TODO: add default x_plt

        # TODO: decorator? better way?
        return vectorize_func(self._prob_single, self.shape)(x)

    def _prob_single(self, x):
        pass

    def plot_prob(self, x=None, ax=None, **kwargs):
        """
        Plot the probability function.

        Parameters
        ----------
        x : array_like, optional
            Random element domain values.
        ax : matplotlib.axes.Axes, optional
            Axes.
        kwargs : dict, optional
            Additional keyword arguments for `self.space.plot` method.

        Returns
        -------
        matplotlib.artist.Artist or tuple of matplotlib.artist.Artist

        """
        return self.space.plot(self.prob, x, ax, **kwargs)

    def sample(self, size=None, rng=None):
        """
        Randomly generate elements.

        Parameters
        ----------
        size : int or tuple, optional
            Number or shape of set of random elements generated.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        np.ndarray
            Array of random elements.

        """
        if size is None:
            shape = ()
        elif isinstance(size, Integral | np.integer):
            shape = (size,)
        elif isinstance(size, tuple):
            shape = size
        else:
            raise TypeError("Input 'size' must be int or tuple.")

        rng = self._get_rng(rng)
        samples = self._sample(math.prod(shape), rng)
        # TODO: use np.asscalar if possible?
        return samples.reshape(shape + samples.shape[1:])

    @abstractmethod
    def _sample(self, n, rng):
        """
        Randomly generate elements, core functionality.

        Parameters
        ----------
        n : int
            Number of elements to generate.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        np.ndarray
            Array of random elements.

        """
        raise NotImplementedError("Method must be overwritten.")


class MixinRV:
    """Mixin class for random variables (numeric domain)."""

    _mean: float | np.ndarray | None
    _cov: float | np.ndarray | None

    # mean = property(lambda self: self._mean)
    # cov = property(lambda self: self._cov)

    @property
    def mean(self):
        """First moment."""
        return self._mean

    @property
    def cov(self):
        """Second central moment."""
        return self._cov

    def expectation(self, f):
        return self.space.integrate(lambda x: self.prob(x) * f(x))


class BaseRV(MixinRV, Base, ABC):
    """
    Base class for random variables (numeric).

    Parameters
    ----------
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, rng=None):
        super().__init__(rng)

        # self._mean = None
        # self._cov = None


class Deterministic(Base):
    """
    Deterministic random element.

    Parameters
    ----------
    value : array_like
        The deterministic value.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    # TODO: redundant, just use FiniteGeneric?
    # TODO: change to ContinuousRV for integration?
    # TODO: General dirac mix?

    def __new__(cls, value, rng=None):
        if np.issubdtype(np.array(value).dtype, np.number):
            return super().__new__(DeterministicRV)
        else:
            return super().__new__(cls)

    def __init__(self, value, rng=None):
        super().__init__(rng)
        self.value = value

    # Input properties
    @property
    def value(self):
        """The deterministic value."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = np.array(value)
        self._space = spaces.FiniteGeneric(self._value, shape=self._value.shape)

        self._mode = self._value

    def _sample(self, n, rng):
        return np.broadcast_to(self._value, (n, *self.shape))

    def prob(self, x):
        return np.where(
            np.all(x.reshape(-1, self.size) == self._value.flatten(), axis=-1), 1.0, 0.0
        )


class DeterministicRV(MixinRV, Deterministic):
    """Deterministic random variable."""

    # @property
    # def value(self):
    #     return self.value

    @Deterministic.value.setter  # type: ignore
    # @value.setter
    def value(self, value):
        # super(DeterministicRV, self.__class__).value.fset(self, value)
        Deterministic.value.fset(self, value)

        self._mean = self._value
        self._cov = np.zeros(2 * self.shape)


class FiniteGeneric(Base):
    """
    Finite-domain random element with specified domain and probabilities.

    Parameters
    ----------
    values : array_like
        Explicit domain values.
    p : array_like, optional
        Probabilities for each value in the domain. Defaults to uniform.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    space: spaces.FiniteGeneric

    # TODO: DRY - use stat approx from the FiniteGeneric space's methods?

    def __new__(cls, values, p=None, rng=None):
        if np.issubdtype(np.array(values).dtype, np.number):
            return super().__new__(FiniteGenericRV)
        else:
            return super().__new__(cls)

    def __init__(self, values, p=None, rng=None):
        super().__init__(rng)

        values = np.array(values)

        if p is None:
            size_p = values.shape[0]
            p = np.ones(size_p) / size_p
        else:
            p = np.array(p)

        self._space = spaces.FiniteGeneric(values, shape=values.shape[p.ndim :])
        self.p = p

    def __eq__(self, other):
        if isinstance(other, FiniteGeneric):
            return np.all(self.values == other.values) and np.all(self.p == other.p)
        return NotImplemented

    def __deepcopy__(self, memodict=None):
        # if memodict is None:
        #     memodict = {}
        return type(self)(self.values, self.p, self.rng)

    def __repr__(self):
        return f"FiniteGeneric(values={self.values}, p={self.p})"

    @classmethod
    def from_grid(cls, lims, n=100, endpoint=True, p=None, rng=None):
        """
        Create random variable with a finite grid of domain values.

        Parameters
        ----------
        lims : array_like
            Lower and upper limits for each dimension.
        n : int, optional
            Number of points defining the grid.
        endpoint : bool, optional
            If True, the upper limit values are included in the grid.
        p : array_like, optional
            Probabilities for each value. Defaults to uniform.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        FiniteGeneric

        """
        space = spaces.FiniteGeneric.from_grid(lims, n, endpoint)
        if p is None:
            p = np.ones(space.set_shape) / space.set_size

        return cls(space.values, p, rng)

    # Input properties
    @property
    def values(self):
        """Domain values."""
        return self.space.values

    @property
    def _values_flat(self):
        return self.space.values_flat

    @property
    def p(self):
        """Probability mass values."""
        return self._p

    @p.setter
    def p(self, p):
        self._p = check_valid_pmf(p)
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        if self._p.shape != self.space.set_shape:
            raise ValueError(f"Shape of 'p' must be {self.space.set_shape}.")
        self._p_flat = self._p.flatten()

        self._mode = None
        # self._mode = self._values_flat[np.argmax(self._p_flat)]

    @property
    def mode(self):
        if self._mode is None:
            self._mode = self._values_flat[np.argmax(self._p_flat)]

        return self._mode

    def _sample(self, n, rng):
        return rng.choice(self._values_flat, size=n, p=self._p_flat)

    def _prob_single(self, x):
        eq = np.all(x == self._values_flat, axis=tuple(range(1, 1 + self.ndim)))
        # eq = np.empty(self.space.set_size, dtype=np.bool)
        # for i, value in enumerate(self._values_flat):
        #     eq[i] = np.allclose(x, value)

        if eq.sum() == 0:
            raise ValueError("Input 'x' must be in the domain.")

        return self._p_flat[eq].squeeze()


class FiniteGenericRV(MixinRV, FiniteGeneric):
    """Finite-domain random variable with specified domain and probabilities."""

    def _update_attr(self):
        super()._update_attr()
        self._mean = None
        self._cov = None

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.tensordot(self._p_flat, self._values_flat, axes=(0, 0))

        return self._mean

    @property
    def cov(self):
        if self._cov is None:
            ctr = self._values_flat - self.mean
            self._cov = sum(
                p_i * np.tensordot(ctr_i, ctr_i, 0)
                for p_i, ctr_i in zip(self._p_flat, ctr)
            )

        return self._cov


class Dirichlet(BaseRV):
    """
    Dirichlet random process, finite-domain realizations.

    Parameters
    ----------
    mean : array_like
    alpha_0 : float
        Localization parameter.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    space: spaces.Simplex

    def __init__(self, mean, alpha_0, rng=None):
        super().__init__(rng)
        self._space = spaces.Simplex(np.array(mean).shape)

        self._alpha_0 = alpha_0
        self._mean = check_valid_pmf(mean, full_support=True)
        self._update_attr()

    def __repr__(self):
        return f"Dirichlet(mean={self.mean}, alpha_0={self.alpha_0})"

    # Input properties
    @property
    def alpha_0(self):
        """Dirichlet localization (i.e. concentration) parameter."""
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = alpha_0
        self._update_attr()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = check_valid_pmf(mean, full_support=True)
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        if np.min(self._mean) > 1 / self._alpha_0:
            self._mode = (self._mean - 1 / self._alpha_0) / (
                1 - self.size / self._alpha_0
            )
        else:
            # warnings.warn("Mode method currently supported for mean > 1/alpha_0 only")
            self._mode = None  # TODO: complete with general formula

        self._cov = (
            np.diagflat(self._mean).reshape(2 * self.shape)
            - np.tensordot(self._mean, self._mean, 0)
        ) / (self._alpha_0 + 1)

        self._log_prob_coef = gammaln(self._alpha_0) - np.sum(
            gammaln(self._alpha_0 * self._mean)
        )

        self.space.x_plt = None

    def _sample(self, n, rng):
        _samps = rng.dirichlet(self._alpha_0 * self._mean.flatten(), size=n)
        return _samps.reshape(n, *self.shape)

    def prob(self, x):
        x, set_shape = check_valid_pmf(x, shape=self.shape)

        if np.logical_and(x == 0, self.mean < 1 / self.alpha_0).any():
            raise ValueError(
                "Each element in 'x' must be greater than "
                "zero if the corresponding mean element is less than 1 / alpha_0."
            )

        log_prob = self._log_prob_coef + np.sum(
            xlogy(self._alpha_0 * self._mean - 1, x).reshape(-1, self.size), -1
        )
        return np.exp(log_prob).reshape(set_shape)

    def plot_prob(self, x=None, ax=None, **kwargs):
        if x is None and self.space._x_plt is None:
            self.space.x_plt = self.space.make_grid(
                self.space.n_plot, self.shape, hull_mask=(self.mean < 1 / self.alpha_0)
            )
        return self.space.plot(self.prob, x, ax)


class Empirical(BaseRV):
    """
    Empirical random process, finite-domain realizations.

    Parameters
    ----------
    mean : array_like
    n : int
        Number of samples characterizing the realized empirical distributions.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, mean, n, rng=None):
        super().__init__(rng)

        self._n = n
        self._mean = check_valid_pmf(mean)

        self._space = spaces.SimplexDiscrete(self.n, self.mean.shape)

        self._update_attr()

    def __repr__(self):
        return f"Empirical(mean={self.mean}, n={self.n})"

    # Input properties
    @property
    def n(self):
        """Number of samples characterizing the realized empirical distributions."""
        return self._n

    @n.setter
    def n(self, n):
        self._n = n
        self._update_attr()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = check_valid_pmf(mean)
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        self._log_prob_coef = gammaln(self._n + 1)

        # self._mode = ((self._n * self._mean) // 1) + simplex_round(
        #     (self._n * self._mean) % 1
        # )  # FIXME: broken
        self._mode = None

        self._cov = (
            np.diagflat(self._mean).reshape(2 * self.shape)
            - np.tensordot(self._mean, self._mean, 0)
        ) / self._n

    @staticmethod
    def simplex_round(x):  # TODO: delete?
        x = np.array(x)
        if np.min(x) < 0:
            raise ValueError("Input values must be non-negative.")
        elif not np.isclose(x.sum(), 1):
            raise ValueError("Input values must sum to one.")

        out = np.zeros(x.size)
        up = 1
        for i, x_i in enumerate(x.flatten()):
            if x_i < up / 2:
                up -= x_i
            else:
                out[i] = 1
                break

        return out.reshape(x.shape)

    def _sample(self, n, rng):
        _samps = rng.multinomial(self._n, self._mean.flatten(), size=n)
        return _samps.reshape(n, *self.shape) / self._n

    def prob(self, x):
        x, set_shape = check_valid_pmf(x, shape=self.shape)
        if (np.minimum((self._n * x) % 1, (-self._n * x) % 1) > 1e-9).any():
            raise ValueError("Each entry in 'x' must be a multiple of 1/n.")

        log_prob = self._log_prob_coef + (
            xlogy(self._n * x, self._mean) - gammaln(self._n * x + 1)
        ).reshape(-1, self.size).sum(axis=-1)
        return np.exp(log_prob).reshape(set_shape)


class DirichletEmpirical(BaseRV):
    """
    Dirichlet-Empirical random process, finite-domain realizations.

    Parameters
    ----------
    mean : array_like
    alpha_0 : float
        Localization parameter.
    n : int
        Number of samples characterizing the realized empirical distributions.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, mean, alpha_0, n, rng=None):
        super().__init__(rng)
        self._space = spaces.SimplexDiscrete(n, np.array(mean).shape)

        self._mean = check_valid_pmf(mean)
        self._alpha_0 = alpha_0
        self._n = n
        self._update_attr()

    def __repr__(self):
        return (
            f"DirichletEmpirical(mean={self.mean}, alpha_0={self.alpha_0}, n={self.n})"
        )

    # Input properties
    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = check_valid_pmf(mean)
        if self._mean.shape != self.shape:
            raise ValueError(f"Mean shape must be {self.shape}.")
        self._update_attr()

    @property
    def alpha_0(self):
        """Dirichlet localization (i.e. concentration) parameter."""
        return self._alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = alpha_0
        self._update_attr()

    @property
    def n(self):
        """Number of samples characterizing the realized empirical distributions."""
        return self._n

    @n.setter
    def n(self, n):
        self._n = n
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        # TODO: mode?

        self._cov = (
            (self._n + self._alpha_0)
            / self._n
            / (1 + self._alpha_0)
            * (
                np.diagflat(self._mean).reshape(2 * self.shape)
                - np.tensordot(self._mean, self._mean, 0)
            )
        )

        self._log_prob_coef = (
            gammaln(self._alpha_0)
            - np.sum(gammaln(self._alpha_0 * self._mean))
            + gammaln(self._n + 1)
            - gammaln(self._alpha_0 + self._n)
        )

    def _sample(self, n, rng):
        theta_flat = rng.dirichlet(self._alpha_0 * self._mean.flatten())
        _samps = rng.multinomial(self._n, theta_flat, size=n)
        return _samps.reshape(n, *self.shape) / self._n

    def prob(self, x):
        x, set_shape = check_valid_pmf(x, shape=self.shape)
        if (np.minimum((self._n * x) % 1, (-self._n * x) % 1) > 1e-9).any():
            raise ValueError("Each entry in 'x' must be a multiple of 1/n.")

        log_prob = self._log_prob_coef + (
            gammaln(self._alpha_0 * self._mean + self._n * x) - gammaln(self._n * x + 1)
        ).reshape(-1, self.size).sum(axis=-1)
        return np.exp(log_prob).reshape(set_shape)


class DirichletEmpiricalScalar(BaseRV):
    """
    Scalar Dirichlet-Empirical random variable.

    Parameters
    ----------
    mean : array_like
    alpha_0 : float
        Localization parameter.
    n : int
        Number of samples characterizing the realized empirical distributions.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    Notes
    -----
    Equivalent to the first element of a 2-dimensional `DirichletEmpirical` random
    variable.

    """

    def __init__(self, mean, alpha_0, n, rng=None):
        super().__init__(rng)

        self._multi = DirichletEmpirical([mean, 1 - mean], alpha_0, n, rng)
        self._space = spaces.FiniteGeneric(np.arange(n + 1) / n)

    def __repr__(self):
        param_strs = (f"{s}={getattr(self, s)}" for s in ["mean", "alpha_0", "n"])
        return f"DirichletEmpiricalScalar({', '.join(param_strs)})"

    # Input properties
    @property
    def mean(self):
        return self._multi.mean[0]

    @mean.setter
    def mean(self, mean):
        self._multi.mean = [mean, 1 - mean]

    @property
    def alpha_0(self):
        """Dirichlet localization (i.e. concentration) parameter."""
        return self._multi.alpha_0

    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._multi.alpha_0 = alpha_0

    @property
    def n(self):
        """Number of samples characterizing the realized empirical distributions."""
        return self._multi.n

    @n.setter
    def n(self, n):
        self._multi.n = n

    # Attribute Updates
    @property
    def cov(self):
        return self._multi.cov[0, 0]

    def _sample(self, n, rng):
        a, b = self.alpha_0 * self._multi.mean
        p = rng.beta(a, b)
        return rng.binomial(self.n, p, size=n) / self.n

    def prob(self, x):
        x = np.array(x)
        return self._multi.prob(np.stack((x, 1 - x), axis=-1))


class Beta(BaseRV):
    """
    Beta random variable.

    Parameters
    ----------
    a : float, optional
        First concentration parameter.
    b : float, optional
        Second concentration parameter.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    Notes
    -----
    Defaults to uniform.

    """

    def __init__(self, a=1.0, b=1.0, rng=None):
        super().__init__(rng)
        self._space = spaces.Box((0, 1))

        if a <= 0 or b <= 0:
            raise ValueError("Parameters must be strictly positive.")
        self._a = a
        self._b = b

        self._update_attr()

    def __repr__(self):
        return f"Beta({self.a}, {self.b})"

    @classmethod
    def from_mean(cls, mean=0.5, alpha_0=2, rng=None):
        """
        Create Beta RV using mean and total concentration.

        Parameters
        ----------
        mean : float, optional
            Mean of the distribution.
        alpha_0 : float, optional
            Total concentration. Conceptually identical to `Dirichlet` parameter.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        Beta

        Notes
        -----
        Defaults to uniform.

        """
        return cls(alpha_0 * mean, alpha_0 * (1 - mean), rng)

    # Input properties
    @property
    def a(self):
        """First concentration parameter."""
        return self._a

    @a.setter
    def a(self, a):
        if a <= 0:
            raise ValueError
        self._a = a
        self._update_attr()

    @property
    def b(self):
        """Second concentration parameter."""
        return self._b

    @b.setter
    def b(self, b):
        if b <= 0:
            raise ValueError
        self._b = b
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        a0 = self._a + self._b

        if self._a > 1:
            if self._b > 1:
                self._mode = (self._a - 1) / (a0 - 2)
            else:
                self._mode = 1
        elif self._a <= 1:
            if self._b > 1:
                self._mode = 0
            elif self._a == 1 and self._b == 1:
                self._mode = 0  # any in unit interval
            else:
                self._mode = 0  # any in {0,1}

        self._mean = self._a / a0
        self._cov = self._a * self._b / a0**2 / (a0 + 1)

    def _sample(self, n, rng):
        return rng.beta(self._a, self._b, size=n)

    def prob(self, x):
        x = np.array(x)
        log_prob = (
            xlog1py(self._b - 1.0, -x)
            + xlogy(self._a - 1.0, x)
            - betaln(self._a, self._b)
        )
        return np.exp(log_prob)


class Binomial(BaseRV):
    """
    Binomial random variable.

    Parameters
    ----------
    p : float
        The probability of the implied Bernoulli RV samples.
    n : int
        The number of implied Bernoulli RV samples.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, p, n, rng=None):
        super().__init__(rng)
        self._space = spaces.FiniteGeneric(np.arange(n + 1))

        if n < 0:
            raise ValueError
        elif p < 0 or p > 1:
            raise ValueError
        self._n = n
        self._p = p

        self._update_attr()

    def __repr__(self):
        return f"Binomial(p={self.p}, n={self.n})"

    # Input properties
    @property
    def n(self):
        """The number of implied Bernoulli RV samples."""
        return self._n

    @n.setter
    def n(self, n):
        if n < 0:
            raise ValueError
        self._n = n
        self._update_attr()

    @property
    def p(self):
        """The probability of the implied Bernoulli RV samples."""
        return self._p

    @p.setter
    def p(self, p):
        if p < 0 or p > 1:
            raise ValueError
        self._p = p
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        temp = (self._n + 1) * self._p
        if temp == 0 or temp % 1 != 0:
            self._mode = math.floor(temp)
        elif temp - 1 in range(self._n):
            self._mode = temp
        elif temp - 1 == self._n:
            self._mode = self._n

        self._mean = self._n * self._p
        self._cov = self._n * self._p * (1 - self._p)

    def _sample(self, n, rng):
        return rng.binomial(self._n, self._p, size=n)

    def prob(self, x):
        x = np.floor(x)
        combiln = gammaln(self._n + 1) - (gammaln(x + 1) + gammaln(self._n - x + 1))
        log_prob = combiln + xlogy(x, self._p) + xlog1py(self._n - x, -self._p)
        return np.exp(log_prob)


class EmpiricalScalar(Binomial):
    """
    Scalar empirical random variable.

    Parameters
    ----------
    mean : float
    n : int
        The number of implied Bernoulli RV samples.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    Notes
    -----
    Equivalent to the first element of a 2-dimensional `Empirical` random variable.

    """

    def __init__(self, mean, n, rng=None):
        super().__init__(mean, n, rng)
        if self.n == 0:
            raise ValueError
        self._space = spaces.FiniteGeneric(np.arange(n + 1) / n)

    def __repr__(self):
        return f"EmpiricalScalar(mean={self.p}, n={self.n})"

    def __eq__(self, other):
        if isinstance(other, EmpiricalScalar):
            return self.n == other.n and self.mean == other.mean
        return NotImplemented

    def _update_attr(self):
        super()._update_attr()
        self._mode /= self._n
        self._mean /= self._n
        self._cov /= self._n**2

    def _sample(self, n, rng):
        return super()._sample(n, rng) / self._n

    def prob(self, x):
        x = np.array(x) * self._n
        return super().prob(x)


class Uniform(BaseRV):
    """
    Uniform random variable over a Box space.

    Parameters
    ----------
    lims : array_like
        Lower and upper limits for each dimension.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    space: spaces.Box

    def __init__(self, lims, rng=None):
        super().__init__(rng)
        self._space = spaces.Box(lims)
        self._update_attr()

    def __repr__(self):
        return f"Uniform({self.lims})"

    # Input properties
    @property
    def lims(self):
        """Lower and upper limits for each dimension."""
        return self.space.lims

    @lims.setter
    def lims(self, value):
        self.space.lims = value
        self._update_attr()

    # Attribute Updates
    def _update_attr(self):
        self._mean = np.mean(self.lims, axis=-1)
        self._mode = self._mean
        _temp = (self.lims[..., 1] - self.lims[..., 0]).flatten() ** 2 / 12
        self._cov = np.diag(_temp).reshape(2 * self.shape)

    def _sample(self, n, rng):
        a_flat = self.lims[..., 0].flatten()
        b_flat = self.lims[..., 1].flatten()
        _temp = np.stack(
            tuple(rng.uniform(a, b, size=n) for a, b in zip(a_flat, b_flat)), axis=-1
        )
        return _temp.reshape((n, *self.shape))

    def prob(self, x):
        x, set_shape = check_data_shape(x, self.shape)
        if not np.all(x >= self.lims[..., 0]) and np.all(x <= self.lims[..., 1]):
            raise ValueError(f"Values must be in interval {self.lims}")

        pr = 1 / np.prod(self.lims[..., 1] - self.lims[..., 0])
        return np.full(set_shape, pr)


class Normal(BaseRV):
    """
    Normal random variable.

    Parameters
    ----------
    mean : float or Collection of float
    cov : float or numpy.ndarray
        Covariance tensor.
    allow_singular : bool, optional
        Whether to allow a singular covariance matrix.
    rng : np.random.Generator or int, optional
        Random number generator seed or object.

    """

    def __init__(self, mean=0.0, cov=1.0, *, allow_singular=True, rng=None):
        super().__init__(rng)
        self.allow_singular = allow_singular

        self._space = spaces.Euclidean(np.array(mean).shape)

        self.mean = mean
        self.cov = cov

    def __repr__(self):
        return f"Normal(mean={self.mean}, cov={self.cov})"

    @property
    def mean(self):
        return self._mean

    @mean.setter
    # @BaseRV.mean.setter
    def mean(self, mean):
        self._mean = np.array(mean)
        if self._mean.shape != self.shape:
            raise ValueError(f"Mean array shape must be {self.shape}.")
        self._mean_flat = self._mean.flatten()

        self._mode = self._mean

        if hasattr(self, "_cov"):
            self._set_lims_plot()  # avoids call before cov is set

    @property
    def cov(self):
        return self._cov

    @cov.setter
    # @BaseRV.cov.setter
    def cov(self, cov):
        self._cov = np.array(cov)

        if self._cov.shape == () and self.ndim == 1:  # TODO: hack-ish?
            self._cov = self._cov * np.eye(self.size)

        if self._cov.shape != self.shape * 2:
            raise ValueError(f"Covariance array shape must be {self.shape * 2}.")
        self._cov_flat = self._cov.reshape(2 * (self.size,))
        # self._cov_flat = self._cov.reshape(2 * self.shape)

        psd = _PSD(self._cov_flat, allow_singular=self.allow_singular)
        self.prec_U = psd.U
        self._log_prob_coef = -0.5 * (psd.rank * np.log(2 * np.pi) + psd.log_pdet)

        self._set_lims_plot()

    def _sample(self, n, rng):
        return rng.multivariate_normal(self._mean_flat, self._cov_flat, size=n).reshape(
            n, *self.shape
        )

    def prob(self, x):
        x, set_shape = check_data_shape(x, self.shape)

        dev = x.reshape(-1, self.size) - self._mean_flat
        maha = np.sum(np.square(np.dot(dev, self.prec_U)), axis=-1)

        log_prob = self._log_prob_coef + -0.5 * maha.reshape(set_shape)
        return np.exp(log_prob)

    def _set_lims_plot(self):
        _n_std = 4
        if self.shape in {(), (2,)}:
            if self.shape == ():
                _margin = _n_std * np.sqrt(self._cov.item())
                lims = self._mean.item() + np.array([-1, 1]) * _margin
            else:  # self.shape == (2,):
                lims = [
                    (
                        self._mean[i] - _n_std * np.sqrt(self._cov[i, i]),
                        self._mean[i] + _n_std * np.sqrt(self._cov[i, i]),
                    )
                    for i in range(2)
                ]

            self._space.lims_plot = lims


class NormalLinear(Normal):
    """
    Normal random variable with mean defined in terms of basis tensors.

    Parameters
    ----------
    weights : array_like
        Weights defining the mean in terms of the basis tensors.
    basis : array_like
        Basis tensors, such that `mean = basis @ weights`.
    cov : float or numpy.ndarray
        Covariance tensor.
    rng : np.random.Generator or int, optional
        Random number generator seed or object.

    """

    # TODO: rework, only allow weights and cov to be set?
    # FIXME: NOT BASIS (incomplete). Rename dictionary?

    def __init__(self, weights=(0.0,), basis=(1.0,), cov=(1.0,), rng=None):
        self._basis = np.array(basis)

        _mean_temp = np.empty(self._basis.shape[:-1])
        super().__init__(_mean_temp, cov, rng=rng)

        self.weights = weights

    def __repr__(self):
        return (
            f"NormalLinear(weights={self.weights}, basis={self.basis}, cov={self.cov})"
        )

    @property
    def weights(self):
        """Weights defining the mean in terms of the basis tensors."""
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = np.array(value)
        if self._weights.ndim != 1:
            raise ValueError("Weights must be 1-dimensional.")
        self.mean = self._basis @ self._weights

    @property
    def basis(self):
        """Basis tensors, such that `mean = basis @ weights`."""
        return self._basis


class DataEmpirical(Base):
    """
    A random element drawn from an empirical distribution.

    Parameters
    ----------
    values : array_like
        The values forming the empirical distribution.
    counts : array_like
        The number of observations for each value.
    space : spaces.Base, optional
        The domain. Defaults to a Euclidean space.
    rng : np.random.Generator or int, optional
        Random number generator seed or object.

    """

    # TODO: subclass for FiniteGeneric space?

    def __new__(cls, values, counts, space=None, rng=None):
        if space is not None:
            dtype = space.dtype
        else:
            dtype = np.array(values).dtype

        if np.issubdtype(dtype, np.number):
            return super().__new__(DataEmpiricalRV)
        else:
            return super().__new__(cls)

    def __init__(self, values, counts, space=None, rng=None):
        super().__init__(rng)

        values, counts = map(np.array, (values, counts))
        if space is None:
            if np.issubdtype(values.dtype, np.number):
                self._space = spaces.Euclidean(values.shape[1:])
            else:
                raise NotImplementedError
        else:
            self._space = space

        self._init_attr()

        self.n = 0
        self.data = self._structure_data([], [])
        self.add_values(values, counts)

    def __repr__(self):
        return f"DataEmpirical(space={self.space}, n={self.n})"

    @classmethod
    def from_data(cls, d, space=None, rng=None):
        """
        Create random element from a dataset.

        Parameters
        ----------
        d : array_like
            The data forming the empirical distribution.
        space : spaces.Base, optional
            The domain. Defaults to a Euclidean space.
        rng : np.random.Generator or int, optional
            Random number generator seed or object.

        Returns
        -------
        DataEmpirical

        """
        values, counts = cls._count_data(d)
        return cls(values, counts, space, rng)

    def _init_attr(self):
        self._mode = np.full(self.shape, np.nan)

    @staticmethod
    def _count_data(d):
        return np.unique(d, return_counts=True, axis=0)

    def _structure_data(self, values, counts):
        return np.array(
            list(zip(values, counts)),
            dtype=[
                ("x", self.dtype, self.shape),
                (
                    "n",
                    np.int32,
                ),
            ],
        )

    def _get_idx(self, x):
        idx = np.flatnonzero(
            np.all(x == self.data["x"], axis=tuple(range(1, 1 + self.ndim)))
        )
        if idx.size == 1:
            return idx.item()
        elif idx.size == 0:
            return None
        else:
            raise ValueError

    def add_data(self, d):
        """
        Add data to the empirical distribution.

        Parameters
        ----------
        d : array_like
            New data added to the empirical distribution.

        """
        self.add_values(*self._count_data(d))

    def add_values(self, values, counts):
        """
        Add new values to the empirical distribution.

        Parameters
        ----------
        values : array_like
            New values for the empirical distribution.
        counts : array_like
            The number of observations for each value.

        """
        values, counts = map(np.array, (values, counts))
        n_new = counts.sum(dtype=np.int32)
        if n_new == 0:
            return

        self.n += n_new

        # Increment existing value counts, flag new values
        idx_new = []
        for i, (value, count) in enumerate(zip(values, counts)):
            idx = self._get_idx(value)
            if idx is not None:
                self.data["n"][idx] += count
            else:
                idx_new.append(i)

        if len(idx_new) > 0:
            self.data = np.concatenate(
                (self.data, self._structure_data(values[idx_new], counts[idx_new]))
            )

        self._update_attr()

    def _update_attr(self):
        self._p = self.data["n"] / self.n

        self._mode = None
        # self._mode = self.data['x'][self.data['n'].argmax()]

        self.space.x_plt = None

    @property
    def mode(self):
        if self._mode is None:
            self._mode = self.data["x"][self.data["n"].argmax()]

        return self._mode

    def _sample(self, size, rng):
        return rng.choice(self.data["x"], size, p=self._p)

    def _prob_single(self, x):
        idx = self._get_idx(x)
        if idx is not None:
            # return self._p[idx]

            # FIXME: delta use needs to account for space dimensionality!?
            if isinstance(self.space, spaces.Continuous):
                delta = 1e250  # approximates Dirac delta function maximal value
            else:
                delta = 1.0

            return self._p[idx] * delta
        else:
            return 0.0

    def plot_prob(self, x=None, ax=None, **kwargs):
        if x is None and self.space.x_plt is None:
            # self.space.set_x_plot()
            if isinstance(self.space, spaces.Continuous) and self.shape in {()}:
                # add empirical values to the plot (so impulses are not missed)
                self.space.x_plt = np.sort(
                    np.unique(np.concatenate((self.space.x_plt, self.data["x"])))
                )

        return self.space.plot(self.prob, x, ax)


class DataEmpiricalRV(MixinRV, DataEmpirical):
    def __repr__(self):
        return f"DataEmpiricalRV(space={self.space}, n={self.n})"

    def _init_attr(self):
        super()._init_attr()

        self._mean = np.full(self.shape, np.nan)
        self._cov = np.full(2 * self.shape, np.nan)

    def _update_attr(self):
        super()._update_attr()

        self._mean = None
        self._cov = None

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.tensordot(self._p, self.data["x"], axes=(0, 0))

        return self._mean

    @property
    def cov(self):
        if self._cov is None:
            ctr = self.data["x"] - self.mean
            self._cov = sum(
                p_i * np.tensordot(ctr_i, ctr_i, 0) for p_i, ctr_i in zip(self._p, ctr)
            )
            # TODO: try np.einsum?

        return self._cov


class Mixture(Base):
    """
    Mixture of random elements.

    Parameters
    ----------
    dists : Collection of Base
        The random elements to be mixed.
    weights : array_like
        The weights combining the random elements.
    rng : np.random.Generator or int, optional
        Random number generator seed or object.

    """

    # TODO: special implementation for FiniteGeneric? get modes, etc?

    def __new__(cls, dists, weights, rng=None):
        if all(isinstance(dist, MixinRV) for dist in dists):
            return super().__new__(MixtureRV)
        else:
            return super().__new__(cls)

    def __init__(self, dists, weights, rng=None):
        super().__init__(rng)
        self._dists = list(dists)

        self._space = spaces.check_spaces(self.dists)

        self.weights = weights

    def __repr__(self):
        _str = "; ".join([f"{prob}: {dist}" for prob, dist in zip(self._p, self.dists)])
        return f"Mixture({_str})"

    dists = property(lambda self: self._dists)
    n_dists = property(lambda self: len(self._dists))

    @property
    def weights(self):
        """The weights combining the random elements."""
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = np.array(value)
        if len(self._weights) != self.n_dists:
            raise ValueError(f"Weights must have length {self.n_dists}.")

        self._update_attr()

    def set_dist_attr(self, idx, **dist_kwargs):
        """
        Set attributes of a distribution.

        Parameters
        ----------
        idx : int
            The distribution index in `dists`.
        dist_kwargs : dict, optional
            Keyword arguments for attribute set.

        """
        dist = self._dists[idx]
        for key, value in dist_kwargs.items():
            setattr(dist, key, value)
        self._update_attr()

    def set_dist(self, idx, dist, weight):  # TODO: type check?
        """
        Set a distribution.

        Parameters
        ----------
        idx : int
            The distribution index to overwrite.
        dist : Base
            The new distribution.
        weight : float
            The new weight.

        """
        self._dists[idx] = dist
        self.weights[idx] = weight
        self._update_attr()  # weights setter not invoked

    @property
    def _idx_nonzero(self):
        """Indices of distributions with non-zero weight."""
        return np.flatnonzero(self._weights)

    def _update_attr(self):
        self._p = self._weights / self._weights.sum()
        self._mode = None
        self.space.x_plt = None

    @property
    def mode(self):
        if self._mode is None:
            if self._idx_nonzero.size == 1:
                self._mode = self._dists[self._idx_nonzero.item()].mode
            else:
                self._mode = self.space.argmax(self.prob)

        return self._mode

    def _sample(self, n, rng):
        idx_rng = rng.choice(self.n_dists, size=n, p=self._p)
        out = np.empty((n, *self.shape), dtype=self.space.dtype)
        for i, dist in enumerate(self.dists):
            idx = np.flatnonzero(i == idx_rng)
            if idx.size > 0:
                out[idx] = dist.sample(size=idx.size)

        return out

    def prob(self, x):
        return sum(self._p[i] * self.dists[i].prob(x) for i in self._idx_nonzero)

    def plot_prob(self, x=None, ax=None, **kwargs):
        if x is None and self.space.x_plt is None:
            # self.space.set_x_plot()

            dists_nonzero = [self.dists[i] for i in self._idx_nonzero]

            # Update space plotting attributes
            if isinstance(self.space, spaces.Euclidean):
                temp = np.stack(list(dist.space.lims_plot for dist in dists_nonzero))
                self._space.lims_plot = [temp[:, 0].min(), temp[:, 1].max()]

            if isinstance(self.space, spaces.Continuous) and self.shape in {()}:
                x = self.space.x_plt
                for dist in dists_nonzero:
                    if isinstance(dist, DataEmpirical):
                        x = np.concatenate((x, dist.data["x"]))

                self.space.x_plt = np.sort(np.unique(x))

        return self.space.plot(self.prob, x, ax)


class MixtureRV(MixinRV, Mixture):
    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"MixtureRV({_str})"

    def _update_attr(self):
        super()._update_attr()

        self._mean = None
        self._cov = None

    @property
    def mean(self):
        if self._mean is None:
            self._mean = sum(self._p[i] * self.dists[i].mean for i in self._idx_nonzero)

        return self._mean
