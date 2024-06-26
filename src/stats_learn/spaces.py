"""Spaces for extrema-finding, integration, plotting, etc."""

import math
from abc import ABC, abstractmethod
from functools import singledispatch
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
from more_itertools import all_equal
from scipy import integrate, optimize

from stats_learn.util import check_data_shape

# TODO: generalize plotting for non-numeric values (e.g. mode_y_x)
# TODO: issubset method?

# TODO: add vectorized `contains` method?


def check_spaces(spaces):
    """
    Check spaces for equivalence.

    Parameters
    ----------
    spaces : Collection
        Spaces to check.

    Returns
    -------
    Base or dict
        A space instance.

    """
    if all_equal(obj.space for obj in spaces):
        return spaces[0].space
    else:
        raise ValueError("All objects must have the same space attribute.")


class Base(ABC):
    """
    Base class for spaces.

    Parameters
    ----------
    shape : tuple
        Shape of the space values.
    dtype : type
        Data type of the space values.

    """

    def __init__(self, shape, dtype):
        self._shape = tuple(shape)
        self._size = math.prod(self._shape)
        self._ndim = len(self._shape)

        self._dtype = dtype

        self._x_plt = None

    @property
    def shape(self):
        """
        Shape of the space values.

        Returns
        -------
        tuple

        """
        return self._shape

    @property  # TODO: `cached_property`?
    def size(self):
        """
        Size of the space values.

        Returns
        -------
        int

        """
        return self._size

    @property
    def ndim(self):
        """
        Dimensionality of the space values.

        Returns
        -------
        int

        """
        return self._ndim

    @property
    def dtype(self):
        """
        Data type of the space values.

        Returns
        -------
        np.dtype

        """
        return self._dtype

    @property
    def x_plt(self):
        """Array of default values for plotting routines."""
        if self._x_plt is None:
            self.set_x_plot()
        return self._x_plt

    @x_plt.setter
    def x_plt(self, value):
        self._x_plt = value

    @abstractmethod
    def set_x_plot(self):
        raise NotImplementedError

    def make_axes(self, **kwargs):
        """Create axes for plotting."""
        with plt.rc_context({"axes.xmargin": 0}):
            if self.shape == ():
                _, ax = plt.subplots(subplot_kw=kwargs)
                ax.set(xlabel="$x$", ylabel="$y$")
            elif self.shape == (2,):
                _kwargs = kwargs | {"projection": "3d"}
                _, ax = plt.subplots(subplot_kw=_kwargs)
                ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$y$")

                ax.set_prop_cycle(
                    "color",
                    [
                        "1f77b4",
                        "ff7f0e",
                        "2ca02c",
                        "d62728",
                        "9467bd",
                        "8c564b",
                        "e377c2",
                        "7f7f7f",
                        "bcbd22",
                        "17becf",
                    ],
                )
            elif self.shape == (3,):
                _kwargs = kwargs | {"projection": "3d"}
                _, ax = plt.subplots(subplot_kw=_kwargs)
                ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$")
            else:
                raise NotImplementedError(
                    "Plotting only supported for 1- and 2- dimensional data."
                )

        return ax

    def plot(self, f, x=None, ax=None, ax_kwargs=None, label=None, **kwargs):
        """
        Plot a function.

        Parameters
        ----------
        f : callable
            The function to plot.
        x : array_like, optional
            Values to plot against. Defaults to `self.x_plt`.
        ax : matplotlib.axes.Axes, optional
            Axes.
        ax_kwargs : dict, optional
            Keyworld arguments for Axes construction.
        label : str, optional
            Label for matplotlib.artist.Artist
        kwargs : dict, optional
            Additional plotting keyword arguments.

        Returns
        -------
        matplotlib.artist.Artist or tuple of matplotlib.artist.Artist

        """
        x, y, set_shape = self._eval_func(f, x)
        return self.plot_xy(x, y, ax=ax, ax_kwargs=ax_kwargs, label=label, **kwargs)

    def plot_xy(
        self,
        x,
        y,
        y_std=None,
        y_std_hi=None,
        ax=None,
        ax_kwargs=None,
        label=None,
        **kwargs,
    ):
        """
        Plot an array.

        Parameters
        ----------
        x : array_like
            Domain (i.e. space) values.
        y : array_like
            Range values.
        y_std : array_like, optional
            Standard deviation for error plotting.
        y_std_hi : array_like, optional
            Upper standard deviation, if different from lower defined in `y_std`.
        ax : matplotlib.axes.Axes, optional
            Axes.
        ax_kwargs : dict, optional
            Keyworld arguments for Axes construction.
        label : str, optional
            Label for matplotlib.artist.Artist
        kwargs : dict, optional
            Additional plotting keyword arguments.

        Returns
        -------
        matplotlib.artist.Artist or tuple of matplotlib.artist.Artist

        """
        if ax is None:
            if ax_kwargs is None:
                ax_kwargs = {}
            ax = self.make_axes(**ax_kwargs)

        x, set_shape = check_data_shape(x, self.shape)
        if y.shape != set_shape:
            raise NotImplementedError

        if len(set_shape) == 1 and self.shape == ():
            kwargs_base = {
                "marker": "." if isinstance(self, Discrete) else "",
                "linestyle": "-",
            }
            kwargs = kwargs_base | kwargs

            plt_data = ax.plot(x, y, label=label, **kwargs)
            if y_std is not None:
                if y_std_hi is None:
                    y_std_hi = y_std

                plt_data_std = ax.fill_between(x, y - y_std, y + y_std_hi, alpha=0.5)
                plt_data = (plt_data, plt_data_std)

        elif len(set_shape) == 1 and self.shape == (2,):
            plt_data = ax.plot(x[..., 0], x[..., 1], y, label=label, **kwargs)

        elif len(set_shape) == 2 and self.shape == (2,):
            plt_data = ax.plot_surface(
                x[..., 0], x[..., 1], y, shade=False, label=label, **kwargs
            )
            plt_data._facecolors2d, plt_data._edgecolors2d = (
                plt_data._facecolor3d,
                plt_data._edgecolor3d,
            )
            # FIXME: use MAYAVI package for 3D??
            # plt_data = ax.plot_surface(x[..., 0], x[..., 1], y, cmap='viridis')
            # plt_data = ax.plot_wireframe(x[..., 0], x[..., 1], y, label=label)

            # if y_std is not None:
            #     if y_std_hi is None:
            #         y_std_hi = y_std

            #     plt_data_lo = ax.plot_surface(
            #         x[..., 0], x[..., 1], y - y_std, shade=False
            #     )
            #     plt_data_hi = ax.plot_surface(
            #         x[..., 0], x[..., 1], y + y_std_hi, shade=False
            #     )
            #     plt_data = (plt_data, (plt_data_lo, plt_data_hi))

            plt_data = [plt_data]

        else:
            raise NotImplementedError

        return plt_data

    def _eval_func(self, f, x=None):
        """Evaluate callable."""
        if x is None:
            x = self.x_plt

        x, set_shape = check_data_shape(x, self.shape)

        y = np.array(f(x))
        if y.shape != set_shape:
            raise ValueError("Function must be scalar-valued.")

        return x, y, set_shape

    # TODO
    @abstractmethod
    def _minimize(self, f):
        raise NotImplementedError

    def argmin(self, f):
        return self._minimize(f)

    def argmax(self, f):
        return self.argmin(lambda x: -1 * f(x))

    # TODO
    @abstractmethod
    def integrate(self, f):
        raise NotImplementedError

    def moment(self, f, order=0, center=False):
        if not np.issubdtype(self.dtype, np.number):
            # TODO: dispatch to subclass with __new__? use mixin?
            raise TypeError("Moments only supported for numeric spaces.")

        if center:
            raise NotImplementedError
        return self.integrate(lambda x: (x**order) * f(x))


class Discrete(Base, ABC):
    """Base class for discrete spaces."""

    pass

    # def plot_xy(
    #     self, x, y, y_std=None, y_std_hi=None, ax=None, label=None, **error_kwargs
    # ):
    #     if ax is None:
    #         ax = self.make_axes()

    #     x, set_shape = check_data_shape(x, self.shape)
    #     if y.shape != set_shape:
    #         raise NotImplementedError

    #     if len(set_shape) == 1 and self.shape == ():
    #         plt_data = ax.plot(x, y, ".", label=label)
    #         if y_std is not None:
    #             if y_std_hi is None:
    #                 y_std_hi = y_std

    #             error_kwargs = {
    #                 "color": ax.lines[-1].get_color(),
    #                 "alpha": 0.5,
    #                 "markersize": 2,
    #             }
    #             ax.plot(x, y - y_std, ".", label=None, **error_kwargs)
    #             ax.plot(x, y + y_std_hi, ".", label=None, **error_kwargs)

    #             # format_kwargs = {'fmt': fmt}
    #             # format_kwargs.update(error_kwargs)
    #             # plt_data = ax.errorbar(x, y, yerr=y_std,
    #             #                        label=label, **format_kwargs)
    #     else:
    #         raise NotImplementedError

    #     return plt_data

    # def plot_xy(
    #     self, x, y, y_std=None, y_std_hi=None, ax=None, label=None, **error_kwargs
    # ):
    #     if ax is None:
    #         ax = self.make_axes()

    #     x, set_shape = check_data_shape(x, self.shape)
    #     if y.shape != set_shape:
    #         raise NotImplementedError

    #     if len(set_shape) == 1 and self.shape == ():
    #         fmt = "."
    #         if y_std is None:
    #             plt_data = ax.plot(x, y, fmt, label=label)
    #         else:
    #             if y_std_hi is not None:
    #                 y_std = np.stack((y_std, y_std_hi))

    #             format_kwargs = {"fmt": fmt}
    #             format_kwargs.update(error_kwargs)
    #             plt_data = ax.errorbar(x, y, yerr=y_std, label=label, **format_kwargs)
    #     else:
    #         raise NotImplementedError

    #     return plt_data


class Finite(Discrete, ABC):
    """Base class for finite spaces."""

    pass


class FiniteGeneric(Finite):
    """
    Finite-dimensional space with specified values.

    Parameters
    ----------
    values : array_like
        Explicit elements of space.
    shape : tuple, optional
        Shape of the space values.

    """

    def __init__(self, values, shape=()):  # TODO: flatten and ignore set shape?
        self.values = np.array(values)
        super().__init__(shape, self.values.dtype)

        _idx_split = self.values.ndim - self.ndim
        if self.values.shape[_idx_split:] != self.shape:
            raise ValueError(f"Support trailing shape must be {self.shape}.")

        self.set_shape = self.values.shape[:_idx_split]
        self.set_size = math.prod(self.set_shape)
        self.set_ndim = len(self.set_shape)

        self._values_flat = self.values.reshape(-1, *self.shape)
        if len(self._values_flat) != len(np.unique(self._values_flat, axis=0)):
            raise ValueError("Input 'values' must have unique values")

    values_flat = property(lambda self: self._values_flat)

    @classmethod
    def from_outer(cls, *vecs):
        """Define support as outer product of tensors."""
        if len(vecs) == 1:
            return cls(vecs, ())
        else:
            grid = np.stack(np.meshgrid(*vecs, indexing="ij"), axis=-1)
            return cls(grid, shape=(len(vecs),))

    @classmethod
    def from_grid(cls, lims, n=100, endpoint=True):
        """Define support as grid over a Box space."""
        grid = Box.make_grid(lims, n, endpoint)
        shape = (grid.shape[-1],) if grid.ndim > 1 else ()
        return cls(grid, shape)

    def __repr__(self):
        return f"FiniteGeneric({self.values})"

    def __eq__(self, other):
        if isinstance(other, FiniteGeneric):
            return self.shape == other.shape and (self.values == other.values).all()
        return NotImplemented

    def __contains__(self, item):
        item = np.array(item)
        if item.shape == self.shape and item.dtype == self.dtype:
            # eq = np.all(item.flatten() == self._values_flat, axis=-1)
            eq = np.all(item == self._values_flat, axis=tuple(range(1, 1 + self.ndim)))
            return eq.sum() > 0
        else:
            return False

    def _minimize(self, f):
        y = np.array(list(map(f, self._values_flat)))
        i_opt = np.argmin(y)
        return self._values_flat[i_opt]

        # # ranges = (np.mgrid[:self.set_size],)
        # ranges = (slice(self.set_size), )
        # i_opt = int(optimize.brute(lambda i: f(self._values_flat[int(i)]), ranges))

        # return self._values_flat[i_opt]

    def integrate(self, f):
        return sum(f(val) for val in self._values_flat)

    def set_x_plot(self):
        self.x_plt = self.values

    # def plot(self, f, x=None, ax=None, label=None, **kwargs):
    #     if ax is None:
    #         ax = self.make_axes()

    #     x, y, set_shape = self._eval_func(f, x)

    #     set_ndim = len(set_shape)
    #     if set_ndim == 1 and self.shape == ():
    #         if np.issubdtype(self.dtype, np.number):
    #             return ax.plot(x, y, ".-", label=label)
    #         else:
    #             return ax.stem(x, y, use_line_collection=True, label=label)

    #     elif (
    #         set_ndim == 2
    #         and self.shape == (2,)
    #         and np.issubdtype(self.dtype, np.number)
    #     ):
    #         # return ax.bar3d(x[..., 0].flatten(), x[..., 1].flatten(),
    #         #                 0, 1, 1, y.flatten(), shade=True)
    #         return ax.plot(
    #             x[..., 0].flatten(),
    #             x[..., 1].flatten(),
    #             y.flatten(),
    #             marker=".",
    #             linestyle="",
    #             label=label,
    #         )

    #     elif (
    #         set_ndim == 3
    #         and self.shape == (3,)
    #         and np.issubdtype(self.dtype, np.number)
    #     ):
    #         plt_data = ax.scatter(
    #             x[..., 0], x[..., 1], x[..., 2], s=15, c=y, label=label
    #         )

    #         c_bar = plt.colorbar(plt_data, ax=ax)
    #         c_bar.set_label("$y$")

    #         return plt_data

    #     else:
    #         raise NotImplementedError(
    #             "Plot method only implemented for 1- and 2- dimensional data."
    #         )


class Continuous(Base, ABC):
    def __init__(self, shape):
        """Abstract base class for continuous spaces."""
        super().__init__(shape, np.float64)

    @property
    def _scipy_kwargs(self):
        return {"x_init": np.zeros(self.shape), "ranges": None}

    def _minimize(self, f):
        # TODO: add args for method and kwargs
        # TODO: add stepsize dependence on lims

        x = self._minimize_brute(f)
        # x = self._minimize_basinhopping(f)

        return x.reshape(self.shape)

    def _minimize_brute(self, f):
        _brute_kwargs = dict(
            ranges=self._scipy_kwargs["ranges"],
            Ns=10,
            # workers=1,
            # finish=None,
            # finish=optimize.minimize
        )
        x = optimize.brute(f, **_brute_kwargs)

        lims = _brute_kwargs.get("ranges")
        if lims is not None:
            a_min, a_max = zip(*lims)
            x = np.clip(x, a_min, a_max)

        return x

    def _minimize_basinhopping(self, f):
        _basin_kwargs = dict(
            x0=self._scipy_kwargs["x_init"],
            niter=100,
            T=1.0,
            stepsize=4.0,
            minimizer_kwargs=dict(bounds=self._scipy_kwargs["ranges"]),
        )
        opt_result = optimize.basinhopping(f, **_basin_kwargs)
        return opt_result.x

    def integrate(self, f):
        y_shape = np.array(f(self._scipy_kwargs["x_init"])).shape
        ranges = self._scipy_kwargs["ranges"]
        if y_shape == ():
            result, *_ = integrate.nquad(lambda *args: f(np.array(args)), ranges)
            return result
        else:
            out = np.empty(self.size)

            def _make_func(i):
                return lambda *args: f(np.array(args))[i]

            for i in range(self.size):
                out[i] = integrate.nquad(_make_func(i), ranges)[0]
                # out[i] = integrate.nquad(lambda *args: f(list(args))[i], ranges)[0]

            return out

        # if self.shape == ():
        #     return integrate.quad(f, *self.lims)[0]
        # if self.shape == (2,):
        #     return integrate.dblquad(lambda x, y: f([x, y]),
        #                              *self.lims[1], *self.lims[0])[0]
        # else:
        #     raise NotImplementedError        # TODO


class Box(Continuous):  # TODO: make Box inherit from Euclidean?
    """
    Orthotope space.

    Parameters
    ----------
    lims : array_like
        Lower and upper limits for each dimension.

    """

    def __init__(self, lims):
        self.lims = lims
        super().__init__(shape=self.lims.shape[:-1])

    def __repr__(self):
        return f"Box({self.lims})"

    def __eq__(self, other):
        if isinstance(other, Box):
            return (self.lims == other.lims).all()
        return NotImplemented

    def __contains__(self, item):
        item = np.array(item)
        if item.shape == self.shape and item.dtype == self.dtype:
            conditions = [
                (item >= self.lims[..., 0]).all(),
                (item <= self.lims[..., 1]).all(),
            ]
            return all(conditions)
        else:
            return False

    @property
    def _scipy_kwargs(self):  # bounds reshaped for scipy optimizer
        return {
            "x_init": self.lims.mean(-1),
            "ranges": self.lims.reshape(-1, 2),
        }

    @property
    def lims(self):
        return self._lims

    @lims.setter
    def lims(self, value):
        self._lims = np.array(value)

        if self.lims.shape[-1] != 2:
            raise ValueError("Trailing shape must be (2,)")
        elif not np.all(self.lims[..., 0] <= self.lims[..., 1]):
            raise ValueError("Upper values must meet or exceed lower values.")

        self._x_plt = None

    @property
    def lims_plot(self):
        return self._lims

    @staticmethod
    def make_grid(lims, n=100, endpoint=True):
        """
        Make a equally-spaced grid of tensors.

        Parameters
        ----------
        lims : array_like
            Lower and upper limits for each dimension.
        n : int, optional
            Number of points defining the plot grid.
        endpoint : bool, optional
            If True, the upper limit values are included in the grid.

        Returns
        -------
        numpy.ndarray

        """
        lims = np.array(lims)
        # if not (lims[..., 0] <= lims[..., 1]).all():
        #     raise ValueError("Upper values must meet or exceed lower values.")

        if lims.shape == (2,):
            return np.linspace(*lims, n, endpoint=endpoint)
        elif lims.ndim == 2 and lims.shape[-1] == 2:
            x_dim = [np.linspace(*lims_i, n, endpoint=endpoint) for lims_i in lims]
            return np.stack(np.meshgrid(*x_dim, indexing="ij"), axis=-1)
        else:
            raise ValueError("Shape must be (2,) or (*, 2)")

    def set_x_plot(self):
        n_plt = 1000 if self.ndim == 0 else 100
        self.x_plt = self.make_grid(self.lims_plot, n_plt, endpoint=False)


class Euclidean(Box):
    """
    A Euclidean space.

    Parameters
    ----------
    shape : tuple
        Shape of the space values.

    """

    def __init__(self, shape):
        if isinstance(shape, Integral | np.integer):
            shape = (shape,)

        lims = np.broadcast_to([-np.inf, np.inf], (*shape, 2))
        super().__init__(lims)

        self._lims_plot = np.broadcast_to([0, 1], shape=(*shape, 2))

    def __repr__(self):
        return f"Euclidean{self.shape}"

    def __eq__(self, other):
        if isinstance(other, Euclidean):
            return self.shape == other.shape
        return NotImplemented

    def __contains__(self, item):
        item = np.array(item)
        if item.shape == self.shape and item.dtype == self.dtype:
            return True
        else:
            return False

    @property
    def _scipy_kwargs(self):  # bounds reshaped for scipy optimizer
        return {
            "x_init": self.lims_plot.mean(-1),
            "ranges": self.lims_plot.reshape(-1, 2),
        }

    @property
    def lims_plot(self):
        return self._lims_plot

    @lims_plot.setter
    def lims_plot(self, value):
        self._lims_plot = np.broadcast_to(value, shape=(*self.shape, 2))
        self._x_plt = None


class Simplex(Continuous):
    """
    A simplex space.

    Parameters
    ----------
    shape : tuple
        Shape of the space values.

    """

    # TODO: add integration and mode finding

    def __init__(self, shape):
        super().__init__(shape)
        self._n_plot = 40

    def __repr__(self):
        return f"Simplex{self.shape}"

    def __eq__(self, other):
        if isinstance(other, Simplex):
            return self.shape == other.shape
        return NotImplemented

    def __contains__(self, item):
        item = np.array(item)
        if item.shape == self.shape and item.dtype == self.dtype:
            conditions = (
                (item >= np.zeros(self.shape)).all(),
                (item <= np.ones(self.shape)).all(),
                np.allclose(item.sum(), 1.0, rtol=1e-9),
            )
            return all(conditions)
        else:
            return False

    @property
    def n_plot(self):
        """Number of points defining the default plot grid."""
        return self._n_plot

    @n_plot.setter
    def n_plot(self, value):
        self._n_plot = value
        self._x_plt = None

    @staticmethod
    def make_grid(n, shape, hull_mask=None):
        """
        Make a equally-spaced grid of tensors.

        Parameters
        ----------
        n : int, optional
            Number of points defining the plot grid.
        shape : tuple
            Shape of the space values.
        hull_mask : array_like, optional
            Where True, defines boundaries to exclude.

        Returns
        -------
        numpy.ndarray

        """
        if not isinstance(n, int) or n < 1:
            raise TypeError("Input 'n' must be a positive integer")

        if type(shape) is not tuple:
            raise TypeError("Input 'shape' must be a tuple of integers.")
        elif not all([isinstance(x, int) for x in shape]):
            raise TypeError("Elements of 'shape' must be integers.")

        d = math.prod(shape)

        if hull_mask is None:
            hull_mask = np.broadcast_to(False, (d,))
        else:
            hull_mask = np.asarray(hull_mask)
            if hull_mask.shape != shape:
                raise TypeError("Input 'hull_mask' must have same shape.")
            elif not all([isinstance(x, np.bool_) for x in hull_mask.flatten()]):
                raise TypeError("Elements of 'hull_mask' must be boolean.")
            hull_mask = hull_mask.flatten()

        if n < sum(hull_mask.flatten()):
            raise ValueError(
                "'n' must meet or exceed the number of True values in 'hull_mask'."
            )

        if d == 1:
            return np.array(1).reshape(shape)

        s = 1 if hull_mask[0] else 0
        e = 0 if (d == 2 and hull_mask[1]) else 1
        g = np.arange(s, n + e)[:, np.newaxis]

        for i in range(1, d - 1):
            s = 1 if hull_mask[i] else 0
            e = 0 if (i == d - 2 and hull_mask[i + 1]) else 1

            g_new = []
            for v in g:
                for k in np.arange(s, n + e - v.sum()):
                    g_new.append(np.append(v, k))
            g = np.array(g_new)

        g = np.hstack((g, n - g.sum(axis=1)[:, np.newaxis]))

        return g.reshape((-1,) + shape) / n

    def set_x_plot(self):
        self.x_plt = self.make_grid(self.n_plot, self._shape)

    def make_axes(self, **kwargs):
        if self.shape == (2,):
            _, ax = plt.subplots(subplot_kw=kwargs)
            ax.set(xlabel="$x_1$", ylabel="$x_2$")
            return ax
        elif self.shape == (3,):
            _kwargs = kwargs | {"projection": "3d"}
            _, ax = plt.subplots(subplot_kw=_kwargs)
            ax.view_init(35, 45)
            ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$")
            return ax
        else:
            raise NotImplementedError(
                "Plot method only supported for 2- and 3-dimensional data."
            )

    def plot(self, f, x=None, ax=None, ax_kwargs=None, label=None, **scatter_kwargs):
        if ax is None:
            if ax_kwargs is None:
                ax_kwargs = {}
            ax = self.make_axes(**ax_kwargs)

        x, y, set_shape = self._eval_func(f, x)
        if len(set_shape) != 1:
            raise ValueError()

        scatter_kwargs = {"label": label, "s": 5, "c": y} | scatter_kwargs

        if self.shape == (2,):
            plt_data = ax.scatter(x[:, 0], x[:, 1], **scatter_kwargs)
        elif self.shape == (3,):
            plt_data = ax.scatter(x[:, 0], x[:, 1], x[:, 2], **scatter_kwargs)
        else:
            raise NotImplementedError(
                "Plot method only supported for 2- and 3-dimensional data."
            )

        c_bar = plt.colorbar(plt_data, ax=ax)
        c_bar.set_label("$y$")

        return plt_data

    def _minimize(self, f):
        raise NotImplementedError

    def integrate(self, f):
        raise NotImplementedError


class SimplexDiscrete(Simplex):  # TODO: bad inheritance from `Continuous`
    """
    Finite grid over Simplex space.

    Parameters
    ----------
    n : int
        Number of points defining the grid.
    shape : tuple
        Shape of the space values.

    """

    def __init__(self, n, shape):
        self.n = n
        super().__init__(shape)
        self.n_plot = self.n

    def __repr__(self):
        return f"SimplexDiscrete({self.n}, {self.shape})"

    def __eq__(self, other):
        if isinstance(other, SimplexDiscrete):
            return self.shape == other.shape and self.n == other.n
        return NotImplemented

    def __contains__(self, item):
        item = np.array(item)
        if item.shape == self.shape and item.dtype == self.dtype:
            conditions = (
                (item >= np.zeros(self.shape)).all(),
                (item <= np.ones(self.shape)).all(),
                np.allclose(item.sum(), 1.0, rtol=1e-9),
                (np.minimum((self.n * item) % 1, (-self.n * item) % 1) < 1e-9).all(),
            )
            return all(conditions)
        else:
            return False

    def _minimize(self, f):
        y = np.array(list(map(f, self.x_plt)))
        i_opt = np.argmin(y)
        return self.x_plt[i_opt]

    def integrate(self, f):
        return sum(f(val) for val in self.x_plt)


@singledispatch
def convex_closure(space: Base):
    """Make convex closure of space.

    Parameters
    ----------
    space : Base
    """
    if isinstance(space, FiniteGeneric):
        vals = space.values_flat
        lims = vals.min(axis=0), vals.max(axis=0)
        space = Box(lims)


@convex_closure.register
def _convex_closure_finite_generic(space: FiniteGeneric):
    vals = space.values_flat
    lims = vals.min(axis=0), vals.max(axis=0)
    return Box(lims)


@convex_closure.register
def _convex_closure_simplex_discrete(space: SimplexDiscrete):
    return Simplex(space.shape)
