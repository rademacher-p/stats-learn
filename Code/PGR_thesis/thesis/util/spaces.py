import math
from functools import partial

import numpy as np
from scipy import optimize, integrate
from matplotlib import pyplot as plt

from thesis.util.plotting import simplex_grid, box_grid
from thesis.util.base import check_data_shape
from thesis.util import plotting

# plt.style.use('seaborn')


# TODO: generalize plotting for non-numeric values (e.g. mode_y_x)
# TODO: issubset method?

def check_spaces(iter_):
    space = iter_[0].space
    if all(obj.space == space for obj in iter_[1:]):
        return space
    else:
        raise ValueError("All objects must have the same space attribute.")


class Space:
    def __init__(self, shape, dtype):
        self._shape = tuple(shape)
        self._size = math.prod(self._shape)
        self._ndim = len(self._shape)

        self._dtype = dtype

        self.x_plt = None
        self.set_x_plot()

    shape = property(lambda self: self._shape)
    size = property(lambda self: self._size)
    ndim = property(lambda self: self._ndim)

    dtype = property(lambda self: self._dtype)

    def set_x_plot(self, x=None):
        pass

    def make_axes(self):        # TODO: axes kwargs
        if self.shape == ():
            _, ax = plt.subplots()
            ax.set(xlabel='$x$', ylabel='$f(x)$')
            return ax
        elif self.shape == (2,):
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f(x)$')
            return ax
        # elif self.shape == (3,):
        #     _, ax = plt.subplots(subplot_kw={'projection': '3d'})
        #     ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')
        #     return ax
        else:
            raise NotImplementedError('Plotting only supported for 1- and 2- dimensional data.')

    def plot(self, f, x=None, ax=None, label=None):
        raise Exception

    def _eval_func(self, f, x=None):
        if x is None:
            x = self.x_plt

        x, set_shape = check_data_shape(x, self.shape)

        y = np.array(f(x))
        if y.shape != set_shape:
            raise ValueError("Function must be scalar-valued.")

        return x, y, set_shape

    def _minimize(self, f):
        pass    # TODO

    def argmin(self, f):
        return self._minimize(f)

    def argmax(self, f):
        return self.argmin(lambda x: -1*f(x))

    def integrate(self, f):
        pass    # TODO

    def moment(self, f, order=1, center=False):
        if not np.issubdtype(self.dtype, np.number):    # TODO: dispatch to subclass with __new__? use mixin?
            raise TypeError("Moments only supported for numeric spaces.")

        if order == 1 and not center:
            return self.integrate(lambda x: np.array(x) * f(x))
        else:
            raise NotImplementedError        # TODO


class Discrete(Space):
    pass


class Finite(Discrete):
    pass


#%%

# class Grid(Finite):
#     def __init__(self, *vecs):
#         # self.vecs = list(map(lambda v: np.sort(np.array(v, dtype=np.float).flatten()), vecs))
#         self.vecs = tuple(np.sort(np.array(vec, dtype=np.float).flatten()) for vec in vecs)
#         super().__init__((len(self.vecs),), np.float)
#
#     def __repr__(self):
#         return f"Grid({self.vecs})"
#
#     def __eq__(self, other):
#         if isinstance(other, Grid):
#             return self.vecs == other.vecs
#         return NotImplemented
#
#     def __contains__(self, item):
#         return all(x_i in vec for x_i, vec in zip(item, self.vecs))
#
#     def set_x_plot(self, x=None):
#         if x is None:
#             self.x_plt = plotting.mesh_grid(*self.vecs)
#         else:
#             self.x_plt = np.array(x)
#
#     def plot(self, f, x=None, ax=None, label=None):
#         if ax is None:
#             ax = self.make_axes()
#
#         x, y, set_shape = self._eval_func(f, x)
#
#         set_ndim = len(set_shape)
#         if set_ndim == 1 and self.shape == ():
#             return ax.stem(x, y, use_line_collection=True, label=label)
#
#         elif set_ndim == 2 and self.shape == (2,):
#             return ax.bar3d(x[..., 0].flatten(), x[..., 1].flatten(), 0, 1, 1, y.flatten(), shade=True)
#
#         elif set_ndim == 3 and self.shape == (3,):
#             plt_data = ax.scatter(x[..., 0], x[..., 1], x[..., 2], s=15, c=y, label=label)
#
#             c_bar = plt.colorbar(plt_data)
#             c_bar.set_label('$f(x)$')
#
#             return plt_data
#
#         else:
#             raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')


class FiniteGeneric(Finite):
    def __init__(self, values, shape=()):       # TODO: flatten and ignore set shape?
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
            raise ValueError("Input 'support' must have unique values")

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
        # ranges = (np.mgrid[:self.set_size],)
        ranges = (slice(self.set_size), )
        i_opt = int(optimize.brute(lambda i: f(self._vals_flat[int(i)]), ranges))

        return self._vals_flat[i_opt]

    def integrate(self, f):
        # y_flat = f(self.values.reshape(-1, *self.shape))     # shouldn't require vectorized funcs?
        y_flat = np.stack([f(val) for val in self._vals_flat])
        return y_flat.sum(0)

    def set_x_plot(self, x=None):
        if x is None:
            self.x_plt = self.values
        else:
            self.x_plt = np.array(x)

    def plot(self, f, x=None, ax=None, label=None):
        if ax is None:
            ax = self.make_axes()

        x, y, set_shape = self._eval_func(f, x)

        set_ndim = len(set_shape)
        if set_ndim == 1 and self.shape == ():
            return ax.stem(x, y, use_line_collection=True, label=label)

        elif set_ndim == 2 and self.shape == (2,):
            return ax.bar3d(x[..., 0].flatten(), x[..., 1].flatten(), 0, 1, 1, y.flatten(), shade=True)

        elif set_ndim == 3 and self.shape == (3,):
            plt_data = ax.scatter(x[..., 0], x[..., 1], x[..., 2], s=15, c=y, label=label)

            c_bar = plt.colorbar(plt_data)
            c_bar.set_label('$f(x)$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')


#%%
class Continuous(Space):
    def __init__(self, shape):
        super().__init__(shape, np.float64)

    @property
    def optimize_kwargs(self):
        return {'x0': np.zeros(self.shape), 'bounds': None, 'constraints': ()}

    def _minimize(self, f):     # TODO: add `brute` method option?
        kwargs = self.optimize_kwargs.copy()
        x0 = kwargs.pop('x0')

        # FIXME: add stepsize dependence on lims
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


class Box(Continuous):      # TODO: make Box inherit from Euclidean?
    def __init__(self, lims):
        self.lims = np.array(lims)

        if self.lims.shape[-1] != 2:
            raise ValueError("Trailing shape must be (2,)")
        elif not (self.lims[..., 0] <= self.lims[..., 1]).all():
            raise ValueError("Upper values must meet or exceed lower values.")

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
    def optimize_kwargs(self):      # bounds reshaped for scipy optimizer
        return {'x0': self.lims_plot.mean(-1), 'bounds': self.lims.reshape(-1, 2), 'constraints': ()}

    @property
    def lims_plot(self):
        return self.lims

    def set_x_plot(self, x=None):   # TODO: simplify?
        if x is None:
            self.x_plt = box_grid(self.lims_plot, 100, endpoint=False)
            # if self.shape in {(), (2,)}:
            #     self.x_plt = box_grid(self.lims_plot, 100, endpoint=False)
            # else:
            #     self.x_plt = None
        else:
            self.x_plt = np.array(x)

    def plot(self, f, x=None, ax=None, label=None):
        if ax is None:
            ax = self.make_axes()

        x, y, set_shape = self._eval_func(f, x)

        set_ndim = len(set_shape)
        if set_ndim == 1 and self.shape == ():
            return ax.plot(x, y, label=label)

        elif set_ndim == 2 and self.shape == (2,):
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f(x)$')

            # return ax.plot_wireframe(x[..., 0], x[..., 1], y)
            return ax.plot_surface(x[..., 0], x[..., 1], y, cmap=plt.cm.viridis)

        else:
            raise NotImplementedError('Plot method only supported for 1- and 2-dimensional data.')


class Euclidean(Box):
    def __init__(self, shape):
        lims = np.broadcast_to([-np.inf, np.inf], (*shape, 2))
        # self._lims_plot = np.array([-1, 1])  # defaults
        self._lims_plot = np.broadcast_to([-1, 1], shape=(*shape, 2))
        super().__init__(lims)

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

    # @property
    # def optimize_kwargs(self):
    #     kwargs = super().optimize_kwargs
    #     kwargs['bounds'] = None
    #     return kwargs

    @property
    def lims_plot(self):
        return self._lims_plot

    @lims_plot.setter
    def lims_plot(self, val):
        self._lims_plot = np.broadcast_to(val, shape=(*self.shape, 2))
        super().set_x_plot()
        # self.set_x_plot()

    def set_x_plot(self, x=None):   # TODO: simplify?
        super().set_x_plot(x)
        temp = self.x_plt.reshape(-1, *self.shape)
        self._lims_plot = np.array([temp.min(0), temp.max(0)])


#%%

# TODO: add integration and mode finding

class Simplex(Continuous):
    def __init__(self, shape):
        super().__init__(shape)

        # self.set_x_plot()
        self.n_plot = 40

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
        self.set_x_plot()

    def set_x_plot(self, x=None):
        if x is None:
            if self.shape in {(2,), (3,)}:
                self.x_plt = simplex_grid(self.n_plot, self._shape)
            else:
                self.x_plt = None
        else:
            self.x_plt = np.array(x)

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

    def plot(self, f, x=None, ax=None, label=None):
        if ax is None:
            ax = self.make_axes()

        x, y, set_shape = self._eval_func(f, x)

        # pf_plt.sum() / (n_plt ** (self._size - 1))

        if len(set_shape) != 1:
            raise ValueError()

        if self.shape == (2,):
            plt_data = ax.scatter(x[:, 0], x[:, 1], s=15, c=y, label=label)
        elif self.shape == (3,):
            plt_data = ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=15, c=y, label=label)
        else:
            raise NotImplementedError('Plot method only supported for 2- and 3-dimensional data.')

        c_bar = plt.colorbar(plt_data)
        c_bar.set_label('$f(x)$')

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
