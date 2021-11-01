from abc import ABC
import math
from numbers import Integral

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize, integrate
from more_itertools import all_equal

from stats_learn.util import check_data_shape


# TODO: generalize plotting for non-numeric values (e.g. mode_y_x)
# TODO: issubset method?

def check_spaces(iter_):
    if all_equal((obj.space for obj in iter_)):
        return iter_[0].space
    else:
        raise ValueError("All objects must have the same space attribute.")


def check_spaces_x(iter_):
    if all_equal((obj.space['x'] for obj in iter_)):
        return iter_[0].space['x']
    else:
        raise ValueError("All objects must have the same space attribute.")


class Base(ABC):
    def __init__(self, shape, dtype):
        self._shape = tuple(shape)
        self._size = math.prod(self._shape)
        self._ndim = len(self._shape)

        self._dtype = dtype

        self._x_plt = None

    shape = property(lambda self: self._shape)
    size = property(lambda self: self._size)
    ndim = property(lambda self: self._ndim)

    dtype = property(lambda self: self._dtype)

    @property
    def x_plt(self):
        if self._x_plt is None:
            self.set_x_plot()
        return self._x_plt

    @x_plt.setter
    def x_plt(self, val):
        self._x_plt = val

    def set_x_plot(self):
        pass

    def make_axes(self):  # TODO: axes kwargs
        if self.shape == ():
            _, ax = plt.subplots()
            ax.set(xlabel='$x$', ylabel='$y$')
        elif self.shape == (2,):
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$y$')

            ax.set_prop_cycle('color', ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2',
                                        '7f7f7f', 'bcbd22', '17becf'])
        elif self.shape == (3,):
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')
        else:
            raise NotImplementedError('Plotting only supported for 1- and 2- dimensional data.')

        return ax

    def plot(self, f, x=None, ax=None, label=None, **kwargs):
        raise Exception

    def plot_xy(self, x, y, y_std=None, y_std_hi=None, ax=None, label=None):
        if ax is None:
            ax = self.make_axes()

        x, set_shape = check_data_shape(x, self.shape)
        if y.shape != set_shape:
            raise NotImplementedError

        if len(set_shape) == 1 and self.shape == ():
            fmt = '-'
            # fmt = '.-' if isinstance(self, Discrete) else '-'  # TODO: remove?

            plt_data = ax.plot(x, y, fmt, label=label)
            if y_std is not None:
                if y_std_hi is None:
                    y_std_hi = y_std

                plt_data_std = ax.fill_between(x, y - y_std, y + y_std_hi, alpha=0.5)
                plt_data = (plt_data, plt_data_std)

        elif len(set_shape) == 2 and self.shape == (2,):
            plt_data = ax.plot_surface(x[..., 0], x[..., 1], y, shade=False, label=label)
            plt_data._facecolors2d, plt_data._edgecolors2d = plt_data._facecolor3d, plt_data._edgecolor3d
            # FIXME: use MAYAVI package for 3D??
            # plt_data = ax.plot_surface(x[..., 0], x[..., 1], y, cmap='viridis')
            # plt_data = ax.plot_wireframe(x[..., 0], x[..., 1], y, label=label)

            # if y_std is not None:
            #     if y_std_hi is None:
            #         y_std_hi = y_std
            #
            #     plt_data_lo = ax.plot_surface(x[..., 0], x[..., 1], y - y_std, shade=False)
            #     plt_data_hi = ax.plot_surface(x[..., 0], x[..., 1], y + y_std_hi, shade=False)
            #     plt_data = (plt_data, (plt_data_lo, plt_data_hi))

        else:
            raise NotImplementedError

        return plt_data

    def _eval_func(self, f, x=None):
        if x is None:
            x = self.x_plt

        x, set_shape = check_data_shape(x, self.shape)

        y = np.array(f(x))
        if y.shape != set_shape:
            raise ValueError("Function must be scalar-valued.")

        return x, y, set_shape

    def _minimize(self, f):
        pass  # TODO

    def argmin(self, f):
        return self._minimize(f)

    def argmax(self, f):
        return self.argmin(lambda x: -1 * f(x))

    def integrate(self, f):
        pass  # TODO

    def moment(self, f, order=1, center=False):
        if not np.issubdtype(self.dtype, np.number):  # TODO: dispatch to subclass with __new__? use mixin?
            raise TypeError("Moments only supported for numeric spaces.")

        if order == 1 and not center:
            return self.integrate(lambda x: np.array(x) * f(x))
        else:
            raise NotImplementedError  # TODO


class Discrete(Base, ABC):
    pass

    # def plot_xy(self, x, y, y_std=None, y_std_hi=None, ax=None, label=None, **error_kwargs):
    #     if ax is None:
    #         ax = self.make_axes()
    #
    #     x, set_shape = check_data_shape(x, self.shape)
    #     if y.shape != set_shape:
    #         raise NotImplementedError
    #
    #     if len(set_shape) == 1 and self.shape == ():
    #         plt_data = ax.plot(x, y, '.', label=label)
    #         if y_std is not None:
    #             if y_std_hi is None:
    #                 y_std_hi = y_std
    #
    #             error_kwargs = {'color': ax.lines[-1].get_color(), 'alpha': .5, 'markersize': 2}
    #             ax.plot(x, y - y_std, '.', label=None, **error_kwargs)
    #             ax.plot(x, y + y_std_hi, '.', label=None, **error_kwargs)
    #
    #             # format_kwargs = {'fmt': fmt}
    #             # format_kwargs.update(error_kwargs)
    #             # plt_data = ax.errorbar(x, y, yerr=y_std, label=label, **format_kwargs)
    #     else:
    #         raise NotImplementedError
    #
    #     return plt_data

    # def plot_xy(self, x, y, y_std=None, y_std_hi=None, ax=None, label=None, **error_kwargs):
    #     if ax is None:
    #         ax = self.make_axes()
    #
    #     x, set_shape = check_data_shape(x, self.shape)
    #     if y.shape != set_shape:
    #         raise NotImplementedError
    #
    #     if len(set_shape) == 1 and self.shape == ():
    #         fmt = '.'
    #         if y_std is None:
    #             plt_data = ax.plot(x, y, fmt, label=label)
    #         else:
    #             if y_std_hi is not None:
    #                 y_std = np.stack((y_std, y_std_hi))
    #
    #             format_kwargs = {'fmt': fmt}
    #             format_kwargs.update(error_kwargs)
    #             plt_data = ax.errorbar(x, y, yerr=y_std, label=label, **format_kwargs)
    #     else:
    #         raise NotImplementedError
    #
    #     return plt_data


class Finite(Discrete, ABC):
    pass


class FiniteGeneric(Finite):
    def __init__(self, values, shape=()):  # TODO: flatten and ignore set shape?
        self.values = np.array(values)
        super().__init__(shape, self.values.dtype)

        _supp_shape = self.values.shape
        _idx_split = self.values.ndim - self.ndim
        if _supp_shape[_idx_split:] != self.shape:
            raise ValueError(f"Support trailing shape must be {self.shape}.")

        self.set_shape = _supp_shape[:_idx_split]
        self.set_size = math.prod(self.set_shape)
        self.set_ndim = len(self.set_shape)

        self._vals_flat = self.values.reshape(-1, *self.shape)
        if len(self._vals_flat) != len(np.unique(self._vals_flat, axis=0)):
            raise ValueError("Input 'values' must have unique values")

    values_flat = property(lambda self: self._vals_flat)

    @classmethod
    def from_outer(cls, *vecs):
        if len(vecs) == 1:
            return cls(vecs, ())
        else:
            grid = np.stack(np.meshgrid(*vecs, indexing='ij'), axis=-1)
            return cls(grid, shape=(len(vecs),))

    @classmethod
    def from_grid(cls, lims, n=100, endpoint=True):
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
            # eq_supp = np.all(item.flatten() == self._vals_flat, axis=-1)
            eq_supp = np.all(item == self._vals_flat, axis=tuple(range(1, 1 + self.ndim)))
            return eq_supp.sum() > 0
        else:
            return False

    def _minimize(self, f):
        i_opt = np.argmin(f(self._vals_flat))
        return self._vals_flat[i_opt]

        # # ranges = (np.mgrid[:self.set_size],)
        # ranges = (slice(self.set_size), )
        # i_opt = int(optimize.brute(lambda i: f(self._vals_flat[int(i)]), ranges))
        #
        # return self._vals_flat[i_opt]

    def integrate(self, f):
        y_flat = f(self._vals_flat)
        return y_flat.sum(0)

    def set_x_plot(self):
        self.x_plt = self.values

    def plot(self, f, x=None, ax=None, label=None, **kwargs):
        if ax is None:
            ax = self.make_axes()

        x, y, set_shape = self._eval_func(f, x)

        set_ndim = len(set_shape)
        if set_ndim == 1 and self.shape == ():
            if np.issubdtype(self.dtype, np.number):
                return ax.plot(x, y, '.-', label=label)
            else:
                return ax.stem(x, y, use_line_collection=True, label=label)

        elif set_ndim == 2 and self.shape == (2,) and np.issubdtype(self.dtype, np.number):
            # return ax.bar3d(x[..., 0].flatten(), x[..., 1].flatten(), 0, 1, 1, y.flatten(), shade=True)
            return ax.plot(x[..., 0].flatten(), x[..., 1].flatten(), y.flatten(), marker='.', linestyle='', label=label)

        elif set_ndim == 3 and self.shape == (3,) and np.issubdtype(self.dtype, np.number):
            plt_data = ax.scatter(x[..., 0], x[..., 1], x[..., 2], s=15, c=y, label=label)

            c_bar = plt.colorbar(plt_data, ax=ax)
            c_bar.set_label('$y$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')


# %%
class Continuous(Base, ABC):
    def __init__(self, shape):
        super().__init__(shape, np.float64)

    @property
    def optimize_kwargs(self):
        return {'x0': np.zeros(self.shape), 'bounds': None, 'constraints': ()}

    def _minimize(self, f):  # TODO: add `brute` method option?

        # FIXME: add stepsize dependence on lims
        kwargs = self.optimize_kwargs.copy()
        x0 = kwargs.pop('x0')
        return optimize.basinhopping(f, x0, niter=100, T=1., stepsize=4., minimizer_kwargs=kwargs).x.reshape(self.shape)

        # if self.ndim == 0:
        #     return optimize.minimize_scalar(f, bounds=self.optimize_kwargs['bounds'])
        # elif self.ndim == 1:
        #     return optimize.minimize(f, **self.optimize_kwargs)
        # else:
        #     raise ValueError

    def integrate(self, f):
        y_shape = f(self.optimize_kwargs['x0']).shape
        ranges = self.optimize_kwargs['bounds']
        if y_shape == ():
            return integrate.nquad(lambda *args: f(list(args)), ranges)[0]
        else:
            out = np.empty(self.size)
            for i in range(self.size):
                out[i] = integrate.nquad(lambda *args: f(list(args))[i], ranges)[0]

            return out

        # if self.shape == ():
        #     return integrate.quad(f, *self.lims)[0]
        # if self.shape == (2,):
        #     return integrate.dblquad(lambda x, y: f([x, y]), *self.lims[1], *self.lims[0])[0]
        # else:
        #     raise NotImplementedError        # TODO


class Box(Continuous):  # TODO: make Box inherit from Euclidean?
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
            return (item >= self.lims[..., 0]).all() and (item <= self.lims[..., 1]).all()
        else:
            return False

    @property
    def optimize_kwargs(self):  # bounds reshaped for scipy optimizer
        return {'x0': self.lims_plot.mean(-1), 'bounds': self.lims.reshape(-1, 2), 'constraints': ()}

    @property
    def lims(self):
        return self._lims

    @lims.setter
    def lims(self, val):
        self._lims = np.array(val)

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
        lims = np.array(lims)
        # if not (lims[..., 0] <= lims[..., 1]).all():
        #     raise ValueError("Upper values must meet or exceed lower values.")

        if lims.shape == (2,):
            return np.linspace(*lims, n, endpoint=endpoint)
        elif lims.ndim == 2 and lims.shape[-1] == 2:
            x_dim = [np.linspace(*lims_i, n, endpoint=endpoint) for lims_i in lims]
            return np.stack(np.meshgrid(*x_dim, indexing='ij'), axis=-1)
        else:
            raise ValueError("Shape must be (2,) or (*, 2)")

    def set_x_plot(self):
        n_plt = 1000 if self.ndim == 0 else 100
        self.x_plt = self.make_grid(self.lims_plot, n_plt, endpoint=False)

    def plot(self, f, x=None, ax=None, label=None, **kwargs):
        x, y, set_shape = self._eval_func(f, x)
        return self.plot_xy(x, y, ax=ax, label=label)


class Euclidean(Box):
    def __init__(self, shape):
        if isinstance(shape, (Integral, np.integer)):
            shape = (shape,)

        lims = np.broadcast_to([-np.inf, np.inf], (*shape, 2))
        super().__init__(lims)

        self._lims_plot = np.broadcast_to([-1, 1], shape=(*shape, 2))  # defaults

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
    def optimize_kwargs(self):
        kwargs = super().optimize_kwargs
        kwargs['bounds'] = None
        return kwargs

    @property
    def lims_plot(self):
        return self._lims_plot

    @lims_plot.setter
    def lims_plot(self, val):
        self._lims_plot = np.broadcast_to(val, shape=(*self.shape, 2))
        self._x_plt = None


# %%

# TODO: add integration and mode finding

class Simplex(Continuous):
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
            conditions = ((item >= np.zeros(self.shape)).all(),
                          (item <= np.ones(self.shape)).all(),
                          np.allclose(item.sum(), 1., rtol=1e-9),
                          )
            return all(conditions)
        else:
            return False

    @property
    def n_plot(self):
        return self._n_plot

    @n_plot.setter
    def n_plot(self, val):
        self._n_plot = val
        self._x_plt = None

    @staticmethod
    def make_grid(n, shape, hull_mask=None):
        """
        Generate a uniform grid over a simplex.
        """

        if type(n) is not int or n < 1:
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
            raise ValueError("Input 'n' must meet or exceed the number of True values in 'hull_mask'.")

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

    def make_axes(self):
        if self.shape == (2,):
            _, ax = plt.subplots()
            ax.set(xlabel='$x_1$', ylabel='$x_2$')
            return ax
        elif self.shape == (3,):
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.view_init(35, 45)
            ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')
            return ax
        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')

    def plot(self, f, x=None, ax=None, label=None, **scatter_kwargs):
        if ax is None:
            ax = self.make_axes()

        x, y, set_shape = self._eval_func(f, x)
        if len(set_shape) != 1:
            raise ValueError()

        kwargs = {'label': label, 's': 5, 'c': y} | scatter_kwargs

        if self.shape == (2,):
            plt_data = ax.scatter(x[:, 0], x[:, 1], **kwargs)
        elif self.shape == (3,):
            plt_data = ax.scatter(x[:, 0], x[:, 1], x[:, 2], **kwargs)
        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')

        c_bar = plt.colorbar(plt_data, ax=ax)
        c_bar.set_label('$y$')

        return plt_data


class SimplexDiscrete(Simplex):
    def __init__(self, n, shape):
        self.n = n
        super().__init__(shape)

    def __repr__(self):
        return f"SimplexDiscrete({self.n}, {self.shape})"

    def __eq__(self, other):
        if isinstance(other, SimplexDiscrete):
            return self.shape == other.shape and self.n == other.n
        return NotImplemented

    def __contains__(self, item):
        item = np.array(item)
        if item.shape == self.shape and item.dtype == self.dtype:
            conditions = ((item >= np.zeros(self.shape)).all(),
                          (item <= np.ones(self.shape)).all(),
                          np.allclose(item.sum(), 1., rtol=1e-9),
                          (np.minimum((self.n * item) % 1, (-self.n * item) % 1) < 1e-9).all(),
                          )
            return all(conditions)
        else:
            return False

    @property
    def n_plot(self):
        return self.n
