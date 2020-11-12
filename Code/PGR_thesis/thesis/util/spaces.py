import math

import numpy as np
from scipy import optimize, integrate
from matplotlib import pyplot as plt

from thesis.util.plot import simplex_grid, box_grid
from thesis.util.generic import check_data_shape

# plt.style.use('seaborn')


# TODO: use get_axes_xy?

# TODO: numerical approximation of mode, etc.?
# TODO: issubset method?

# def infer_space(sample):      # TODO
#     sample = np.array(sample)
#     if np.issubdtype(sample.dtype, np.number):
#         return Euclidean(sample.shape)
#     else:
#         pass


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
    # size = property(lambda self: math.prod(self._shape))
    # ndim = property(lambda self: len(self._shape))

    dtype = property(lambda self: self._dtype)

    def set_x_plot(self, x=None):
        pass

    def _eval_func(self, f, x=None):
        if x is None:
            x = self.x_plt

        x, set_shape = check_data_shape(x, self.shape)

        y = np.array(f(x))
        if y.shape != set_shape:
            raise ValueError("Function must be scalar-valued.")

        return x, y, set_shape

    # def argmax(self, f, x=None):        # FIXME: implement with scipy.optimize!!
    #     x, y, set_shape = self._eval_func(f, x)
    #
    #     x_flat, y_flat = x.reshape(-1, *self.shape), y.flatten()
    #     return x_flat[np.argmax(y_flat)]

    # def integrate(self, f, x=None):   # TODO: use scipy.integrate
    #     pass        # TODO: also, need delta, not n_lim for approx? Confirm PDF sum to one.
    #
    # def moment(self, f, x=None, order=1, central=False):
    #     pass    # TODO:


class Discrete(Space):
    pass


class Finite(Discrete):
    pass


class FiniteGeneric(Finite):
    def __init__(self, support, shape=()):
        self.support = np.array(support)
        super().__init__(shape, self.support.dtype)

        _supp_shape = self.support.shape
        _idx_split = self.support.ndim - self.ndim
        if _supp_shape[_idx_split:] != self.shape:
            raise ValueError(f"Support trailing shape must be {self.shape}.")

        self._supp_flat = self.support.reshape(-1, self.size)
        if len(self._supp_flat) != len(np.unique(self._supp_flat, axis=0)):
            raise ValueError("Input 'support' must have unique values")

        self.set_shape = _supp_shape[:_idx_split]
        self.set_size = math.prod(self.set_shape)
        self.set_ndim = len(self.set_shape)

        # self.set_x_plot()

    def __repr__(self):
        return f"FiniteGeneric({self.support})"

    def __eq__(self, other):
        if isinstance(other, FiniteGeneric):
            return self.shape == other.shape and (self.support == other.support).all
        return NotImplemented

    def __contains__(self, item):
        item = np.array(item)
        if item.shape == self.shape and item.dtype == self.dtype:
            eq_supp = np.all(item.flatten() == self._supp_flat, axis=-1)
            return eq_supp.sum() > 0
        else:
            return False

    # def argmax(self, f):      # TODO
    #     x, y, set_shape = self._eval_func(f, self.support)
    #
    #     x_flat, y_flat = x.reshape(-1, *self.shape), y.flatten()
    #     return x_flat[np.argmax(y_flat)]

    def _minimize(self, f):
        def g(i):
            x_ = self._supp_flat[int(i)].reshape(self.shape)
            return f(x_)

        return optimize.brute(g, (np.mgrid[:self.set_size],))

    def argmin(self, f):
        i = self._minimize(f)
        return self._supp_flat[int(i)].reshape(self.shape)
        # return self._minimize(f).x.reshape(self.shape)

    def argmax(self, f):
        return self.argmin(lambda x: -1*f(x))

    def set_x_plot(self, x=None):
        if x is None:
            self.x_plt = self.support
        else:
            self.x_plt = np.array(x)

    def plot(self, f, x=None, ax=None):
        x, y, set_shape = self._eval_func(f, x)     # TODO: exception if set_shape != self.set_shape?

        set_ndim = len(set_shape)
        if set_ndim == 1 and self.shape == ():
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x$', ylabel='$f(x)$')

            plt_data = ax.stem(x, y, use_line_collection=True)

            return plt_data

        elif set_ndim == 2 and self.shape == (2,):
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f(x)$')

            plt_data = ax.bar3d(x[..., 0].flatten(), x[..., 1].flatten(), 0, 1, 1, y.flatten(), shade=True)
            return plt_data

        elif set_ndim == 3 and self.shape == (3,):
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

            plt_data = ax.scatter(x[..., 0], x[..., 1], x[..., 2], s=15, c=y)

            c_bar = plt.colorbar(plt_data)
            c_bar.set_label('$f(x)$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')


#
class Continuous(Space):
    def __init__(self, shape):
        super().__init__(shape, np.float64)

    @property
    def optimize_kwargs(self):
        return {'x0': np.zeros(self.shape), 'bounds': None, 'constraints': ()}

    def _minimize(self, f):
        kwargs = self.optimize_kwargs.copy()
        x0 = kwargs.pop('x0')
        return optimize.basinhopping(f, x0, minimizer_kwargs=kwargs)        # TODO: add `brute` method option?
        # if self.ndim == 0:
        #     return optimize.minimize_scalar(f, bounds=self.optimize_kwargs['bounds'])
        # elif self.ndim == 1:
        #     return optimize.minimize(f, **self.optimize_kwargs)
        # else:
        #     raise ValueError

    def argmin(self, f):
        return self._minimize(f).x.reshape(self.shape)

    def argmax(self, f):
        return self.argmin(lambda x: -1*f(x))


class Box(Continuous):
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
    def optimize_kwargs(self):
        x0 = self.lims_plot.sum(-1).reshape(self.size) / 2
        return {'x0': x0, 'bounds': self.lims.reshape(-1, 2), 'constraints': ()}

    # @property
    # def optimize_kwargs(self):
    #     x0 = self.lims_plot.sum(-1) / 2
    #     return {'x0': x0, 'bounds': self.lims, 'constraints': ()}

    @property
    def lims_plot(self):
        return self.lims

    def set_x_plot(self, x=None):   # TODO: simplify?
        if x is None:
            if self.shape in {(), (2,)}:
                # lims = np.broadcast_to([0, 1], shape=(*self.shape, 2))      # defaults to unit hypercube
                # self.x_plt = box_grid(lims, 100, endpoint=False)
                self.x_plt = box_grid(self.lims_plot, 100, endpoint=False)
            else:
                self.x_plt = None
        else:
            self.x_plt = np.array(x)

    def plot(self, f, x=None, ax=None):
        x, y, set_shape = self._eval_func(f, x)

        set_ndim = len(set_shape)
        if set_ndim == 1 and self.shape == ():
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x_1$', ylabel='$f(x)$')        # TODO: ax_kwargs

            plt_data = ax.plot(x, y)
            return plt_data

        elif set_ndim == 2 and self.shape == (2,):
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f(x)$')

            # ax.plot_wireframe(x[..., 0], x[..., 1], self.pf(x))
            plt_data = ax.plot_surface(x[..., 0], x[..., 1], y, cmap=plt.cm.viridis)

            return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 1- and 2-dimensional data.')


class Euclidean(Box):
    def __init__(self, shape):
        lims = np.broadcast_to([-np.inf, np.inf], (*shape, 2))
        self._lims_plot = [-1, 1]  # defaults
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

    @property
    def optimize_kwargs(self):
        kwargs = super().optimize_kwargs
        kwargs['bounds'] = None
        return kwargs
        # return {'bounds': None, 'constraints': ()}

    @property
    def lims_plot(self):
        return self._lims_plot

    @lims_plot.setter
    def lims_plot(self, val):
        self._lims_plot = np.broadcast_to(val, shape=(*self.shape, 2))
        self.set_x_plot()

# class Euclidean(Continuous):
#     def __init__(self, shape: tuple):
#         super().__init__(shape)
#
#         # self.set_x_plot()
#         self.lims_plot = [0, 1]     # defaults to unit hypercube
#
#     def __repr__(self):
#         return f"Euclidean{self.shape}"
#
#     def __eq__(self, other):
#         if isinstance(other, Euclidean):
#             return self.shape == other.shape
#         return NotImplemented
#
#     def __contains__(self, item):
#         item = np.array(item)
#         if item.shape == self.shape and item.dtype == self.dtype:
#             return True
#         else:
#             return False
#
#     def _minimize(self, f):
#         x0 = self.lims_plot.sum(-1) / 2
#         return optimize.minimize(f, x0)
#
#     def argmin(self, f):
#         return self._minimize(f).x
#
#     def argmax(self, f):
#         return self.argmin(lambda x: -1*f(x))
#
#     def set_x_plot(self, x=None):   # TODO: simplify?
#         if x is None:
#             if self.shape in {(), (2,)}:
#                 # lims = np.broadcast_to([0, 1], shape=(*self.shape, 2))      # defaults to unit hypercube
#                 # self.x_plt = box_grid(lims, 100, endpoint=False)
#                 self.x_plt = box_grid(self.lims_plot, 100, endpoint=False)
#             else:
#                 self.x_plt = None
#         else:
#             self.x_plt = np.array(x)
#
#     @property
#     def lims_plot(self):
#         return self._lims_plot
#
#     @lims_plot.setter
#     def lims_plot(self, val):
#         self._lims_plot = np.broadcast_to(val, shape=(*self.shape, 2))
#         self.set_x_plot()
#
#     def plot(self, f, x=None, ax=None):
#         x, y, set_shape = self._eval_func(f, x)
#
#         set_ndim = len(set_shape)
#         if set_ndim == 1 and self.shape == ():
#             if ax is None:
#                 _, ax = plt.subplots()
#                 ax.set(xlabel='$x_1$', ylabel='$f(x)$')        # TODO: ax_kwargs
#
#             plt_data = ax.plot(x, y)
#             return plt_data
#
#         elif set_ndim == 2 and self.shape == (2,):
#             if ax is None:
#                 _, ax = plt.subplots(subplot_kw={'projection': '3d'})
#                 ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f(x)$')
#
#             # ax.plot_wireframe(x[..., 0], x[..., 1], self.pf(x))
#             plt_data = ax.plot_surface(x[..., 0], x[..., 1], y, cmap=plt.cm.viridis)
#
#             return plt_data
#
#         else:
#             raise NotImplementedError('Plot method only supported for 1- and 2-dimensional data.')
#
#
# class Box(Euclidean):
#     def __init__(self, lims):
#         self.lims = np.array(lims)
#
#         if self.lims.shape[-1] != 2:
#             raise ValueError("Trailing shape must be (2,)")
#         elif not (self.lims[..., 0] <= self.lims[..., 1]).all():
#             raise ValueError("Upper values must meet or exceed lower values.")
#
#         super().__init__(shape=self.lims.shape[:-1])
#
#         self.lims_plot = self.lims
#
#     def __repr__(self):
#         return f"Box({self.lims})"
#
#     def __eq__(self, other):
#         if isinstance(other, Box):
#             return self.lims == other.lims
#         return NotImplemented
#
#     def __contains__(self, item):
#         item = np.array(item)
#         if item.shape == self.shape and item.dtype == self.dtype:
#             return (item >= self.lims[..., 0]).all() and (item <= self.lims[..., 1]).all()
#         else:
#             return False


#
class Simplex(Continuous):
    def __init__(self, shape):
        super().__init__(shape)

        self.set_x_plot()

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

    def set_x_plot(self, x=None):
        if x is None:
            if self.shape in {(2,), (3,)}:
                self.x_plt = simplex_grid(40, self._shape)
            else:
                self.x_plt = None
        else:
            self.x_plt = np.array(x)

    def plot(self, f, x=None, ax=None):
        x, y, set_shape = self._eval_func(f, x)

        # pf_plt.sum() / (n_plt ** (self._size - 1))

        if len(set_shape) != 1:
            raise ValueError()

        if self.shape == (2,):
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x_1$', ylabel='$x_2$')

            plt_data = ax.scatter(x[:, 0], x[:, 1], s=15, c=y)

        elif self.shape == (3,):
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.view_init(35, 45)
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

            plt_data = ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=15, c=y)
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

    def set_x_plot(self, x=None):
        if x is None:
            if self.shape in {(2,), (3,)}:
                self.x_plt = simplex_grid(self.n, self._shape)
            else:
                self.x_plt = None
        else:
            self.x_plt = np.array(x)
