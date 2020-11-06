import math

import numpy as np
from matplotlib import pyplot as plt

from thesis.util.plot import simplex_grid, box_grid

# TODO: use get_axes_xy?

# TODO: numerical approximation of mode, etc.?

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

    shape = property(lambda self: self._shape)
    size = property(lambda self: self._size)
    ndim = property(lambda self: self._ndim)
    # size = property(lambda self: math.prod(self._shape))
    # ndim = property(lambda self: len(self._shape))

    dtype = property(lambda self: self._dtype)


class Discrete(Space):
    pass


class Finite(Discrete):
    pass


class Continuous(Space):
    def __init__(self, shape):
        super().__init__(shape, np.float64)


#
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
        self.set_ndim = len(self.set_shape)

        self._set_x_plot()

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

    def _set_x_plot(self):
        self.x_plt = self.support

    def plot(self, f, x=None, ax=None):
        if x is None:
            x = self.x_plt
            set_ndim = self.set_ndim
        else:
            x = np.array(x)
            # set_shape = x.shape[:x.ndim - self.ndim]
            set_ndim = x.ndim - self.ndim

        if set_ndim == 1 and self.shape == ():
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x$', ylabel='$f(x)$')

            plt_data = ax.stem(x, f(x), use_line_collection=True)

            return plt_data

        elif set_ndim == 2 and self.shape == (2,):
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f(x)$')

            plt_data = ax.bar3d(x[..., 0].flatten(), x[..., 1].flatten(), 0, 1, 1, f(x).flatten(), shade=True)
            return plt_data

        elif set_ndim == 3 and self.shape == (3,):
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

            plt_data = ax.scatter(x[..., 0], x[..., 1], x[..., 2], s=15, c=f(x))

            c_bar = plt.colorbar(plt_data)
            c_bar.set_label('$f(x)$')

            return plt_data

        else:
            raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')


#
class Euclidean(Continuous):
    def __init__(self, shape: tuple):
        super().__init__(shape)

        self._set_x_plot()

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

    def _set_x_plot(self):
        if self.shape in {(), (2,)}:
            lims = np.broadcast_to([0, 1], shape=(*self.shape, 2))
            self.x_plt = box_grid(lims, 100, endpoint=False)
        else:
            self.x_plt = None

    def plot(self, f, x=None, ax=None):
        if x is None:
            x = self.x_plt

        if self.shape == ():
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x_1$', ylabel='$f(x)$')        # TODO: ax_kwargs

            plt_data = ax.plot(x, f(x))
            return plt_data

        elif self.shape == (2,):
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f(x)$')

            # ax.plot_wireframe(x[..., 0], x[..., 1], self.pf(x))
            plt_data = ax.plot_surface(x[..., 0], x[..., 1], f(x), cmap=plt.cm.viridis)

            return plt_data

        else:
            raise NotImplementedError('Plot method only supported for 1- and 2-dimensional data.')


class Box(Euclidean):
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
            return self.lims == other.lims
        return NotImplemented

    def __contains__(self, item):
        item = np.array(item)
        if item.shape == self.shape and item.dtype == self.dtype:
            return (item >= self.lims[..., 0]).all() and (item <= self.lims[..., 1]).all()
        else:
            return False

    def _set_x_plot(self):
        if self.shape in {(), (2,)}:
            self.x_plt = box_grid(self.lims, 100, endpoint=False)
        else:
            self.x_plt = None


#
class Simplex(Continuous):
    def __init__(self, shape):
        super().__init__(shape)

        self._set_x_plot()

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

    def _set_x_plot(self):
        if self.shape in {(2,), (3,)}:
            self.x_plt = simplex_grid(40, self._shape)
        else:
            self.x_plt = None

    def plot(self, f, x=None, ax=None):
        if x is None:
            x = self.x_plt

        # pf_plt.sum() / (n_plt ** (self._size - 1))

        if self.shape == (2,):
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x_1$', ylabel='$x_2$')

            plt_data = ax.scatter(x[:, 0], x[:, 1], s=15, c=f(x))

        elif self.shape == (3,):
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.view_init(35, 45)
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

            plt_data = ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=15, c=f(x))
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

    def _set_x_plot(self):
        if self.shape in {(2,), (3,)}:
            self.x_plt = simplex_grid(self.n, self._shape)
        else:
            self.x_plt = None
