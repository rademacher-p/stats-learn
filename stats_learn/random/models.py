r"""Random models of jointly distributed :math:`\xrm` and :math:`\yrm` elements for supervised learning."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Optional

import numpy as np

from stats_learn import random, spaces
from stats_learn.util import RandomGeneratorMixin, vectorize_func

# TODO: add marginal/conditional prob methods


class Base(RandomGeneratorMixin, ABC):
    r"""
    Base class for random supervised learning models.

    Parameters
    ----------
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    Notes
    -----
    Implements a joint distribution between an random elements :math:`\xrm` and `\yrm`. For supervised learning, it is
    assumed that the former is observed while the latter is not.

    """
    _space: Dict[str, Optional[spaces.Base]]

    def __init__(self, rng=None):
        super().__init__(rng)
        self._space = {"x": None, "y": None}

        self._model_x = None
        self._mode_x = None

    # TODO: use NumPy structured dtypes to detail type and shape!?!

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

    @property
    def model_x(self):
        r"""Random variable characterizing the marginal distribution of :math:`\xrm`."""
        return self._model_x

    @abstractmethod
    def model_y_x(self, x):
        r"""
        Generate the conditional random element :math:`\yrm` given :math:`x`

        Parameters
        ----------
        x : array_like
            Observed random element value.

        Returns
        -------
        random.elements.Base
            The random element :math:`\yrm` given :math:`x`

        """
        raise NotImplementedError

    # TODO: default stats to reference `model_x` and `model_y_x` attributes?

    @property
    def mode_x(self):
        r"""The most probable value of :math:`\xrm`."""
        return self._mode_x

    def mode_y_x(self, x):
        r"""
        The most probable values of :math:`\yrm`, given :math:`x` values.

        Parameters
        ----------
        x : array_like
            Observed random element values.

        Returns
        -------
        np.ndarray

        """
        return vectorize_func(self._mode_y_x_single, self.shape["x"])(x)

    def _mode_y_x_single(self, x):
        pass

    def plot_mode_y_x(self, x=None, ax=None):
        r"""
        Plot the mode of :math:`\yrm` for different :math:`x` values.

        Parameters
        ----------
        x : array_like, optional
            Observed random element values. Defaults to `self.space.x_plt`.
        ax : matplotlib.axes.Axes, optional
            Axes onto which stats/losses are plotted.

        Returns
        -------
        matplotlib.artist.Artist or tuple of matplotlib.artist.Artist

        """
        return self.space["x"].plot(self.mode_y_x, x, ax)

    sample = random.elements.Base.sample

    def _sample(self, n, rng):
        d_x = np.array(self.model_x.sample(n, rng=rng))
        d_y = np.array([self.model_y_x(x).sample(rng=rng) for x in d_x])

        return np.array(
            list(zip(d_x, d_y)), dtype=[(c, self.dtype[c], self.shape[c]) for c in "xy"]
        )


class MixinRVx:
    r"""Mixin class for observed random variables :math:`\xrm` (numeric)."""

    _mean_x: Optional[np.ndarray]
    _cov_x: Optional[np.ndarray]

    @property
    def mean_x(self):
        r"""First moment of :math:`\xrm`."""
        return self._mean_x

    @property
    def cov_x(self):
        r"""Second central moment of :math:`\xrm`."""
        return self._cov_x


class MixinRVy:
    r"""Mixin class for unobserved random variables :math:`\yrm` (numeric)."""

    space: dict
    shape: dict

    def mean_y_x(self, x):
        r"""
        The first moments of :math:`\yrm`, given :math:`x` values.

        Parameters
        ----------
        x : array_like
            Observed random element values.

        Returns
        -------
        np.ndarray

        """
        return vectorize_func(self._mean_y_x_single, self.shape["x"])(x)

    def _mean_y_x_single(self, x):
        pass

    def plot_mean_y_x(self, x=None, ax=None):
        r"""
        Plot the mean of :math:`\yrm` for different :math:`x` values.

        Parameters
        ----------
        x : array_like, optional
            Observed random element values. Defaults to `self.space.x_plt`.
        ax : matplotlib.axes.Axes, optional
            Axes onto which stats/losses are plotted.

        Returns
        -------
        matplotlib.artist.Artist or tuple of matplotlib.artist.Artist

        """
        return self.space["x"].plot(self.mean_y_x, x, ax)

    def cov_y_x(self, x):
        r"""
        The second central moments of :math:`\yrm`, given :math:`x` values.

        Parameters
        ----------
        x : array_like
            Observed random element values.

        Returns
        -------
        np.ndarray

        """
        return vectorize_func(self._cov_y_x_single, self.shape["x"])(x)

    def _cov_y_x_single(self, x):
        pass

    def plot_cov_y_x(self, x=None, ax=None):
        r"""
        Plot the covariance of :math:`\yrm` for different :math:`x` values.

        Parameters
        ----------
        x : array_like, optional
            Observed random element values. Defaults to `self.space.x_plt`.
        ax : matplotlib.axes.Axes, optional
            Axes onto which stats/losses are plotted.

        Returns
        -------
        matplotlib.artist.Artist or tuple of matplotlib.artist.Artist

        """
        return self.space["x"].plot(self.cov_y_x, x, ax)


class BaseRVx(MixinRVx, Base):
    @abstractmethod
    def model_y_x(self, x):
        raise NotImplementedError


class BaseRVy(MixinRVy, Base):
    @abstractmethod
    def model_y_x(self, x):
        raise NotImplementedError


class BaseRVxy(MixinRVx, MixinRVy, Base):
    @abstractmethod
    def model_y_x(self, x):
        raise NotImplementedError


# class DataConditionalGeneric(Base):
#     def __new__(cls, model_x, model_y_x, rng=None):
#         is_numeric_y = isinstance(model_y_x(model_x.sample()), random.elements.MixinRV)
#         if isinstance(model_x, random.elements.MixinRV):
#             if is_numeric_y:
#                 return super().__new__(DataConditionalRVxy)
#             else:
#                 return super().__new__(DataConditionalRVx)
#         else:
#             if is_numeric_y:
#                 return super().__new__(DataConditionalRVy)
#             else:
#                 return super().__new__(cls)
#
#     def __init__(self, model_x, model_y_x, rng=None):
#         super().__init__(rng)
#         self._model_x = model_x
#         self._space['x'] = self._model_x.space
#         self._update_x()
#
#         self._model_y_x_ = model_y_x
#         self._space['y'] = self.model_y_x(self._model_x.sample()).space
#
#     @property
#     def model_x(self):
#         return self._model_x
#
#     @model_x.setter
#     def model_x(self, model_x):
#         self._model_x = model_x
#         self._update_x()
#
#     def _update_x(self):
#         self._mode_x = self._model_x.mode
#
#     def model_y_x(self, x):
#         return self.model_y_x_(x)
#
#     @property
#     def model_y_x_(self):
#         return self._model_y_x_
#
#     @model_y_x_.setter
#     def model_y_x_(self, model_y_x):
#         self._model_y_x_ = model_y_x
#
#     def _mode_y_x_single(self, x):
#         return self.model_y_x(x).mode
#
#     @classmethod
#     def from_finite(cls, dists, values_x, p_x=None, rng=None):
#         model_x = random.elements.FiniteGeneric(values_x, p_x)
#
#         def model_y_x(x):
#             eq = np.all(x == model_x.space._values_flat, axis=tuple(range(1, 1 + model_x.space.ndim)))
#             idx = np.flatnonzero(eq).item()
#             return dists[idx]
#
#         return cls(model_x, model_y_x, rng)
#
#
# class DataConditionalRVx(MixinRVx, DataConditional):
#     def _update_x(self):
#         super()._update_x()
#         self._mean_x = self._model_x.mean
#         self._cov_x = self._model_x.cov
#
#
# class DataConditionalRVy(MixinRVy, DataConditional):
#     def _mean_y_x_single(self, x):
#         return self.model_y_x(x).mean
#
#     def _cov_y_x_single(self, x):
#         return self.model_y_x(x).cov
#
#
# class DataConditionalRVxy(DataConditionalRVy, DataConditionalRVx):
#     pass


class DataConditional(Base):
    r"""
    Model with a finite-domain random element :math:`\xrm` and explicit conditional distributions of :math:`y`.

    Parameters
    ----------
    dists : Collection of random.elements.Base
        Explicit conditional random elements characterizing :math:`\yrm` for each possible value of :math:`x`.
    model_x : stats_learn.random.elements.Base
        Random variable characterizing the marginal distribution of :math:`\xrm`.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    def __new__(cls, dists, model_x, rng=None):
        is_numeric_y = all(isinstance(dist, random.elements.MixinRV) for dist in dists)
        if isinstance(model_x, random.elements.MixinRV):
            if is_numeric_y:
                return super().__new__(DataConditionalRVxy)
            else:
                return super().__new__(DataConditionalRVx)
        else:
            if is_numeric_y:
                return super().__new__(DataConditionalRVy)
            else:
                return super().__new__(cls)

    def __init__(self, dists, model_x, rng=None):
        super().__init__(rng)

        self._dists = list(dists)
        self._model_x = model_x

        self._space["x"] = self.model_x.space
        if not isinstance(self.space["x"], spaces.FiniteGeneric):
            raise ValueError(f"Data space must be finite.")
        elif self.space["x"].set_shape != (len(self.dists),):
            raise ValueError("Data space set shape must match number of distributions.")

        self._space["y"] = spaces.check_spaces(self.dists)

    @classmethod
    def from_finite(cls, dists, values_x, p_x=None, rng=None):
        r"""
        Constructor for `FiniteGeneric` marginal model of :math:`\xrm`.

        Parameters
        ----------
        dists : Collection of random.elements.Base
            Explicit conditional random elements characterizing :math:`\yrm` for each possible value of :math:`x`.
        values_x : array_like
            Explicit domain values for :math:`x`.
        p_x : array_like, optional
            Probabilities for each value :math:`x` in the domain. Defaults to uniform.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        DataConditional

        """
        model_x = random.elements.FiniteGeneric(values_x, p_x)
        return cls(dists, model_x, rng)

    @classmethod
    def from_mean_emp(cls, alpha_0, n, func, model_x, rng=None):
        r"""
        Construct Dirichlet-Empirical conditional random variables from the conditional mean function.

        Parameters
        ----------
        alpha_0 : float
            Localization parameter.
        n : int
            Number of samples characterizing the realized empirical distributions.
        func : callable
            The conditional mean function.
        model_x : stats_learn.random.elements.Base
            Random variable characterizing the marginal distribution of :math:`\xrm`.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        DataConditional

        Notes
        -----
        If `alpha_0` is infinite, `EmpiricalScalar` distributions are generated instead.

        """
        # if not isinstance(model_x.space, spaces.FiniteGeneric):
        #     raise ValueError("`model_x` must have `FiniteGeneric` space.")
        if np.isinf(alpha_0):
            dists = [random.elements.EmpiricalScalar(func(x), n - 1) for x in model_x.space.values]
        else:
            dists = [
                random.elements.DirichletEmpiricalScalar(func(x), alpha_0, n - 1)
                for x in model_x.space.values
            ]
        return cls(dists, model_x, rng)

    @classmethod
    def from_mean_poly_emp(cls, alpha_0, n, weights, model_x, rng=None):
        r"""
        Construct Dirichlet-Empirical conditional random variables with a polynomial conditional mean function.

        Parameters
        ----------
        alpha_0 : float
            Localization parameter.
        n : int
            Number of samples characterizing the realized empirical distributions.
        weights : array_like
            The weights combining the polynomial functions into the conditional mean function.
        model_x : stats_learn.random.elements.Base
            Random variable characterizing the marginal distribution of :math:`\xrm`.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        DataConditional

        """

        def poly_func(x):
            return sum(w * x**i for i, w in enumerate(weights))

        return cls.from_mean_emp(alpha_0, n, poly_func, model_x, rng)

    def __eq__(self, other):
        if isinstance(other, DataConditional):
            return self.model_x == other.model_x and self.dists == other.dists
        return NotImplemented

    def __deepcopy__(self, memodict=None):
        # if memodict is None:
        #     memodict = {}
        # return type(self)(self.dists, self.model_x)
        return type(self)(deepcopy(self.dists), deepcopy(self.model_x))

    dists = property(lambda self: self._dists)
    model_x = property(lambda self: self._model_x)

    # @property  # TODO: delete? More general, not necessarily `FiniteGeneric` element!
    # def p_x(self):
    #     r"""The marginal probabilities for values :math:`x`."""
    #     return self.model_x.p
    #
    # @p_x.setter
    # def p_x(self, value):
    #     self.model_x.p = value

    @property
    def mode_x(self):
        return self.model_x.mode

    def _get_idx_x(self, x):
        return self.space["x"].values.tolist().index(x)

    def model_y_x(self, x):
        return self.dists[self._get_idx_x(x)]

    def _mode_y_x_single(self, x):
        return self.model_y_x(x).mode


class DataConditionalRVx(MixinRVx, DataConditional):
    model_x: random.elements.BaseRV

    def _get_idx_x(self, x):
        return np.flatnonzero(np.isclose(x, self.space["x"].values)).item()

    @property
    def mean_x(self):
        return self.model_x.mean

    @property
    def cov_x(self):
        return self.model_x.cov


class DataConditionalRVy(MixinRVy, DataConditional):
    def _mean_y_x_single(self, x):
        return self.model_y_x(x).mean

    def _cov_y_x_single(self, x):
        return self.model_y_x(x).cov


class DataConditionalRVxy(DataConditionalRVy, DataConditionalRVx):
    pass


class ClassConditional(MixinRVx, Base):
    r"""
    Model with a finite-domain random element :math:`\yrm` and explicit conditional distributions of :math:`x`.

    Parameters
    ----------
    dists : Collection of random.elements.Base
        Explicit conditional random elements characterizing :math:`\xrm` for each possible value of :math:`y`.
    model_y : stats_learn.random.elements.Base
        Random variable characterizing the marginal distribution of :math:`\yrm`.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    def __init__(self, dists, model_y, rng=None):
        super().__init__(rng)

        self._dists = list(dists)
        self._model_y = model_y

        self._space["y"] = self.model_y.space
        if not isinstance(self.space["y"], spaces.FiniteGeneric):
            raise ValueError(f"Class space must be finite.")
        elif self.space["y"].set_shape != (len(self.dists),):
            raise ValueError("Class space set shape must match number of distributions.")
        # elif not np.issubdtype(self.space['y'].dtype, 'U'):  # TODO: deprecate dtype enforce
        #     raise ValueError("Space must be categorical")

        self._p_y = None

        self._space["x"] = spaces.check_spaces(self.dists)

    @classmethod
    def from_finite(cls, dists, values_y, p_y=None, rng=None):
        r"""
        Constructor for `FiniteGeneric` marginal model of :math:`\yrm`.

        Parameters
        ----------
        dists : Collection of random.elements.Base
            Explicit conditional random elements characterizing :math:`\xrm` for each possible value of :math:`y`.
        values_y : array_like
            Explicit domain values for :math:`y`.
        p_y : array_like, optional
            Probabilities for each value :math:`y` in the domain. Defaults to uniform.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        ClassConditional

        """
        model_y = random.elements.FiniteGeneric(values_y, p_y)  # TODO: shouldn't enforce dtype
        # model_y = random.elements.FiniteGeneric(np.array(values_y, dtype='U').flatten(), p_y)
        return cls(dists, model_y, rng)

    dists = property(lambda self: self._dists)
    model_y = property(lambda self: self._model_y)

    @property
    def p_y(self):
        r"""The marginal probabilities for values :math:`y`."""
        if self._p_y is None:
            self._p_y = self.model_y.prob(self.space["y"].x_plt)
        return self._p_y
        # return self.model_y.p

    # @p_y.setter  # TODO: delete?
    # def p_y(self, value):
    #     self.model_y.p = value
    #     self._update_attr()

    def _update_attr(self):
        self._model_x = None

    def _mode_y_x_single(self, x):
        raise NotImplementedError

    @property
    def model_x(self):
        if self._model_x is None:
            self._model_x = random.elements.MixtureRV(self.dists, self.p_y)

        return self._model_x

    @property
    def mode_x(self):
        return self.model_x.mode

    @property
    def mean_x(self):
        return self.model_x.mean

    @property
    def cov_x(self):
        return self.model_x.cov

    def mode_y_x(self, x):
        temp = np.array([dist.prob(x) * p for dist, p in zip(self.dists, self.p_y)])
        return self.space["y"].values[temp.argmax(axis=0)]

    def model_y_x(self, x):
        temp = np.array([dist.prob(x) * p for dist, p in zip(self.dists, self.p_y)])
        p_y_x = temp / temp.sum()
        return random.elements.FiniteGeneric(self.space["y"].values, p_y_x)

    def model_x_y(self, y):
        idx = self.space["y"].values.tolist().index(y)
        return self.dists[idx]

    def _sample(self, n, rng):
        d_y = np.array(self.model_y.sample(n, rng=rng))
        d_x = np.array([self.model_x_y(y).sample(rng=rng) for y in d_y])

        return np.array(
            list(zip(d_x, d_y)), dtype=[(c, self.dtype[c], self.shape[c]) for c in "xy"]
        )


class BetaLinear(MixinRVx, MixinRVy, Base):
    r"""
    Model characterized by a Beta conditional distribution with mean defined in terms of basis functions.

    Parameters
    ----------
    weights : array_like
        The weights combining the basis functions into the conditional mean function.
    basis_y_x : Collection of callable, optional
        Basis functions. Defaults to polynomial functions.
    alpha_y_x : float, optional
        Total conditional Beta concentration. Defaults to uniform.
    model_x : stats_learn.random.elements.Base
        Random variable characterizing the marginal distribution of :math:`\xrm`.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.
    """

    # TODO: DRY with NormalLinear

    def __init__(
        self,
        weights=(0.0,),
        basis_y_x=None,
        alpha_y_x=2.0,
        model_x=random.elements.Beta(),
        rng=None,
    ):
        super().__init__(rng)

        self._space["x"] = model_x.space
        self._space["y"] = spaces.Box((0, 1))

        self.model_x = model_x

        self.weights = weights
        self.alpha_y_x = alpha_y_x

        if basis_y_x is None:

            def power_func(i):
                return vectorize_func(
                    lambda x: np.full(self.shape["y"], (x**i).mean()),
                    shape=self.shape["x"],
                )

            self._basis_y_x = tuple(power_func(i) for i in range(len(self.weights)))
        else:
            self._basis_y_x = basis_y_x

    def __repr__(self):
        return (
            f"BetaLinear(model_x={self.model_x}, basis_y_x={self.basis_y_x}, "
            f"weights={self.weights}, alpha_y_x={self.alpha_y_x})"
        )

    @property
    def basis_y_x(self):
        """Basis functions."""
        return self._basis_y_x

    @property
    def model_x(self):
        return self._model_x

    @model_x.setter
    def model_x(self, model_x):
        self._model_x = model_x

    @property
    def mode_x(self):
        return self.model_x.mode

    @property
    def mean_x(self):
        return self.model_x.mean

    @property
    def cov_x(self):
        return self.model_x.cov

    def mean_y_x(self, x):
        return sum(weight * func(x) for weight, func in zip(self.weights, self._basis_y_x))

    def cov_y_x(self, x):
        mean = self.mean_y_x(x)
        return mean * (1 - mean) / (self.alpha_y_x + 1)

    def model_y_x(self, x):
        return random.elements.Beta.from_mean(self.mean_y_x(x), self.alpha_y_x)

    def _mode_y_x_single(self, x):
        return self.model_y_x(x).mode


class NormalLinear(MixinRVx, MixinRVy, Base):
    r"""
    Model characterized by a Normal conditional distribution with mean defined in terms of basis functions.

    Parameters
    ----------
    weights : array_like
        The weights combining the basis functions into the conditional mean function.
    basis_y_x : Collection of callable, optional
        Basis functions. Defaults to polynomial functions.
    cov_y_x : float or callable, optional
        Conditional covariance of Normal distributions.
    model_x : stats_learn.random.elements.Base
        Random variable characterizing the marginal distribution of :math:`\xrm`.
    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator seed or object.

    """

    def __init__(
        self,
        weights=(0.0,),
        basis_y_x=None,
        cov_y_x=1.0,
        model_x=random.elements.Normal(),
        rng=None,
    ):
        super().__init__(rng)

        self.model_x = model_x

        self.weights = weights
        self.cov_y_x_ = cov_y_x

        if basis_y_x is None:

            def power_func(i):
                return vectorize_func(
                    lambda x: np.full(self.shape["y"], (x**i).mean()),
                    shape=self.shape["x"],
                )

            self._basis_y_x = tuple(power_func(i) for i in range(len(self.weights)))
        else:
            self._basis_y_x = basis_y_x

    def __repr__(self):
        return (
            f"NormalModel(model_x={self.model_x}, basis_y_x={self.basis_y_x}, "
            f"weights={self.weights}, cov_y_x={self._cov_repr})"
        )

    @property
    def basis_y_x(self):
        """Basis functions."""
        return self._basis_y_x

    @property
    def model_x(self):
        return self._model_x

    @model_x.setter
    def model_x(self, model_x):
        self._model_x = model_x
        self._space["x"] = model_x.space

    @property
    def mode_x(self):
        return self.model_x.mode

    @property
    def mean_x(self):
        return self.model_x.mean

    @property
    def cov_x(self):
        return self.model_x.cov

    @property
    def cov_y_x_(self):
        return self._cov_repr

    @cov_y_x_.setter
    def cov_y_x_(self, value):
        if callable(value):
            self._cov_repr = value
            self._cov_y_x_single = value
            _temp = self._cov_y_x_single(self.model_x.sample()).shape
        else:
            self._cov_repr = np.array(value)
            self._cov_y_x_single = lambda x: self._cov_repr
            _temp = self._cov_repr.shape

        self._space["y"] = spaces.Euclidean(_temp[: int(len(_temp) / 2)])

    def mean_y_x(self, x):
        return sum(weight * func(x) for weight, func in zip(self.weights, self._basis_y_x))

    def mode_y_x(self, x):
        return self.mean_y_x(x)

    def model_y_x(self, x):
        mean = self.mean_y_x(x)
        cov = self._cov_y_x_single(x)
        return random.elements.Normal(mean, cov)


class DataEmpirical(Base):
    r"""
    A random model drawn from an empirical distribution.

    Parameters
    ----------
    values : array_like
        The values forming the empirical distribution.
    counts : array_like
        The number of observations for each value.
    space : dict, optional
        The domain for :math:`\xrm` and :math:`\yrm`. Each defaults to a Euclidean space.
    rng : np.random.Generator or int, optional
        Random number generator seed or object.

    """

    def __new__(cls, values, counts, space=None, rng=None):
        if space is not None:
            dtype = {c: space[c].dtype for c in "xy"}
        else:
            _dtype = np.array(values).dtype
            dtype = {c: _dtype[c].base for c in "xy"}

        if np.issubdtype(dtype["x"], np.number):
            if np.issubdtype(dtype["y"], np.number):
                return super().__new__(DataEmpiricalRVxy)
            else:
                return super().__new__(DataEmpiricalRVx)
        else:
            if np.issubdtype(dtype["y"], np.number):
                return super().__new__(DataEmpiricalRVy)
            else:
                return super().__new__(cls)

    def __init__(self, values, counts, space=None, rng=None):
        super().__init__(rng)

        values, counts = map(np.array, (values, counts))
        if space is None:
            dtype = np.array(values).dtype
            for c in "xy":
                if np.issubdtype(dtype[c].base, np.number):
                    self._space[c] = spaces.Euclidean(dtype[c].shape)
                else:
                    raise NotImplementedError
        else:
            self._space = space

        self._model_x = random.elements.DataEmpirical([], [], space=self.space["x"])
        self._models_y_x = []

        self.n = 0
        self.data = self._structure_data({"x": [], "y": []}, [])
        self.add_values(values, counts)

    def __repr__(self):
        return f"DataEmpirical(space={self.space}, n={self.n})"

    @classmethod
    def from_data(cls, d, space=None, rng=None):
        r"""
        Create random model from a dataset.

        Parameters
        ----------
        d : np.ndarray
            The data forming the empirical distribution.
        space : dict, optional
            The domain for :math:`\xrm` and :math:`\yrm`. Each defaults to a Euclidean space.
        rng : np.random.Generator or int, optional
            Random number generator seed or object.

        Returns
        -------
        DataEmpirical

        """
        return cls(*cls._count_data(d), space, rng)

    @staticmethod
    def _count_data(d):
        return np.unique(d, return_counts=True, axis=0)

    def _structure_data(self, values, counts):
        return np.array(
            list(zip(values["x"], values["y"], counts)),
            dtype=[
                ("x", self.dtype["x"], self.shape["x"]),
                ("y", self.dtype["y"], self.shape["y"]),
                (
                    "n",
                    int,
                ),
            ],
        )

    def add_data(self, d):
        """
        Add data to the empirical distribution.

        Parameters
        ----------
        d : np.ndarray
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
        n_new = counts.sum(dtype=int)
        if n_new == 0:
            return

        self.n += n_new

        # Increment existing value counts, flag new values
        idx_new = []
        for i, (value, count) in enumerate(zip(values, counts)):
            idx = np.flatnonzero(value == self.data[["x", "y"]])
            if idx.size == 1:
                self.data["n"][idx.item()] += count
            else:
                idx_new.append(i)

        if len(idx_new) > 0:
            self.data = np.concatenate(
                (self.data, self._structure_data(values[idx_new], counts[idx_new]))
            )

        _, idx = np.unique(values["x"], axis=0, return_index=True)
        values_x_unique = values["x"][np.sort(idx)]
        for x_add in values_x_unique:
            idx_match = np.flatnonzero(
                np.all(x_add == values["x"], axis=tuple(range(1, 1 + self.ndim["x"])))
            )
            values_y, counts_y = values["y"][idx_match], counts[idx_match]

            idx = self._model_x._get_idx(x_add)
            if idx is None:
                self._models_y_x.append(
                    random.elements.DataEmpirical(values_y, counts_y, self.space["y"])
                )
            else:
                self._models_y_x[idx].add_values(values_y, counts_y)

            self._model_x.add_values([x_add], [counts[idx_match].sum()])

        self._update_attr()

    def _update_attr(self):
        self._p = self.data["n"] / self.n

    @property
    def mode_x(self):
        return self.model_x.mode

    def _get_idx_x(self, x):
        idx = np.flatnonzero(
            np.all(x == self.model_x.data["x"], axis=tuple(range(1, 1 + self.ndim["x"])))
        )
        if idx.size == 1:
            return idx.item()
        elif idx.size == 0:
            return None
        else:
            raise ValueError

    def model_y_x(self, x):
        idx = self._get_idx_x(x)
        if idx is not None:
            return self._models_y_x[idx]
        else:
            return random.elements.DataEmpirical([], [], space=self.space["y"])

    def _mode_y_x_single(self, x):
        idx = self._get_idx_x(x)
        if idx is not None:
            return self._models_y_x[idx].mode
        else:
            return np.nan

    def _sample(self, size, rng):
        return rng.choice(self.data[["x", "y"]], size, p=self._p)


class DataEmpiricalRVx(MixinRVx, DataEmpirical):
    def __repr__(self):
        return f"DataEmpiricalRVx(space={self.space}, n={self.n})"

    @property
    def mean_x(self):
        return self.model_x.mean

    @property
    def cov_x(self):
        return self.model_x.cov


class DataEmpiricalRVy(MixinRVy, DataEmpirical):
    def __repr__(self):
        return f"DataEmpiricalRVy(space={self.space}, n={self.n})"

    def _mean_y_x_single(self, x):
        idx = self._get_idx_x(x)
        if idx is not None:
            return self._models_y_x[idx].mean
        else:
            return np.nan


class DataEmpiricalRVxy(DataEmpiricalRVx, DataEmpiricalRVy):
    def __repr__(self):
        return f"DataEmpiricalRVxy(space={self.space}, n={self.n})"


class Mixture(Base):
    """
    Mixture of random models.

    Parameters
    ----------
    dists : Collection of Base
        The random models to be mixed.
    weights : array_like
        The weights combining the random models.
    rng : np.random.Generator or int, optional
        Random number generator seed or object.

    """

    # TODO: special implementation for FiniteGeneric? get modes, etc?

    def __new__(cls, dists, weights, rng=None):
        if all(isinstance(dist, MixinRVx) for dist in dists):
            if all(isinstance(dist, MixinRVy) for dist in dists):
                return super().__new__(MixtureRVxy)
            else:
                return super().__new__(MixtureRVx)
        else:
            if all(isinstance(dist, MixinRVy) for dist in dists):
                return super().__new__(MixtureRVy)
            else:
                return super().__new__(cls)

    def __init__(self, dists, weights, rng=None):
        super().__init__(rng)
        self._dists = list(dists)

        self._space = spaces.check_spaces(self.dists)

        self.weights = weights

    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"Mixture({_str})"

    def __deepcopy__(self, memodict=None):
        # if memodict is None:
        #     memodict = {}
        return type(self)(self.dists, self.weights, self.rng)

    dists = property(lambda self: self._dists)
    n_dists = property(lambda self: len(self._dists))

    @property
    def weights(self):
        """The weights combining the random models."""
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = np.array(value)
        if len(self._weights) != self.n_dists:
            raise ValueError(f"Weights must have length {self.n_dists}.")

        self._update_attr()

    def set_dist_attr(
        self, idx, **dist_kwargs
    ):  # TODO: improved implementation w/ direct self.dists access?
        """
        Set attributes of a distribution.

        Parameters
        ----------
        idx : int
            The distribution index in `dists`.
        dist_kwargs : dict, optional
            Keyword arguments for attribute set.

        """
        for key, value in dist_kwargs.items():
            setattr(self._dists[idx], key, value)
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
        return np.flatnonzero(self._weights)

    def _update_attr(self):
        self._p = self._weights / self.weights.sum()
        self._model_x = None

    @property
    def model_x(self):
        if self._model_x is None:
            if self._idx_nonzero.size == 1:
                self._model_x = self._dists[self._idx_nonzero.item()].model_x
            else:
                args = zip(*[(self.dists[i].model_x, self.weights[i]) for i in self._idx_nonzero])
                self._model_x = random.elements.Mixture(*args)

        return self._model_x

    @property
    def mode_x(self):
        return self.model_x.mode

    def _mode_y_x_single(self, x):
        return self.model_y_x(x).mode

    def model_y_x(self, x):
        _weights = self._weights_y_x(x)
        idx_nonzero = np.flatnonzero(_weights)
        if idx_nonzero.size == 1:
            return self._dists[idx_nonzero.item()].model_y_x(x)
        else:
            args = zip(*[(self.dists[i].model_y_x(x), _weights[i]) for i in idx_nonzero])
            return random.elements.Mixture(*args)

    def _weights_y_x(self, x):
        return np.array([w * dist.model_x.prob(x) for w, dist in zip(self.weights, self.dists)])

    def _sample(self, n, rng):
        idx_rng = rng.choice(self.n_dists, size=n, p=self._p)
        out = np.array(
            [tuple(np.empty(self.shape[c], self.dtype[c]) for c in "xy") for _ in range(n)],
            dtype=[(c, self.dtype[c], self.shape[c]) for c in "xy"],
        )
        for i, dist in enumerate(self.dists):
            idx = np.flatnonzero(i == idx_rng)
            if idx.size > 0:
                out[idx] = dist.sample(size=idx.size)

        return out


class MixtureRVx(MixinRVx, Mixture):
    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"MixtureRVx({_str})"

    def _update_attr(self):
        super()._update_attr()

        # self._mean_x = sum(self._p[i] * self.dists[i].mean_x for i in self._idx_nonzero)
        self._mean_x = np.nansum([prob * dist.mean_x for prob, dist in zip(self._p, self.dists)])


class MixtureRVy(MixinRVy, Mixture):
    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"MixtureRVy({_str})"

    def mean_y_x(self, x):
        w = self._weights_y_x(x)
        w_sum = w.sum(axis=0)
        p_y_x = np.full(w.shape, np.nan)

        idx = np.nonzero(w_sum)
        for p_i, w_i in zip(p_y_x, w):
            p_i[idx] = w_i[idx] / w_sum[idx]

        # idx_nonzero = np.flatnonzero(p_y_x)
        # return sum(p_y_x[i] * self.dists[i].mean_y_x(x) for i in idx_nonzero)

        temp = np.array([prob * dist.mean_y_x(x) for prob, dist in zip(p_y_x, self.dists)])
        return np.nansum(temp, axis=0)


class MixtureRVxy(MixtureRVx, MixtureRVy):
    def __repr__(self):
        _str = "; ".join([f"{w}: {dist}" for w, dist in zip(self.weights, self.dists)])
        return f"MixtureRVxy({_str})"
