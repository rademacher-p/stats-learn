import math

import numpy as np
from matplotlib import pyplot as plt

from thesis.util.plot import simplex_grid, box_grid


class Space:
    shape = property(lambda self: self._shape)
    size = property(lambda self: math.prod(self._shape))
    ndim = property(lambda self: len(self._shape))


class Discrete(Space):
    pass


class Finite(Discrete):
    pass


class Continuous(Space):
    pass


#

class Euclidean(Continuous):
    def __init__(self, shape):
        self._shape = shape
        self._set_x_plot()

    def _set_x_plot(self):
        if self.shape in ((), (2,)):
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
                ax.set(xlabel='$x_1$', ylabel='$p$')        # TODO: ax_kwargs

            plt_data = ax.plot(x, f(x))
            return plt_data

        elif self.shape == (2,):
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$p$')

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

    def _set_x_plot(self):
        if self.shape in ((), (2,)):
            self.x_plt = box_grid(self.lims, 100, endpoint=False)
        else:
            self.x_plt = None


class Simplex(Continuous):
    def __init__(self, shape):
        self._shape = shape

        self._set_x_plot()

    def _set_x_plot(self):
        if self.shape in ((2,), (3,)):
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
        c_bar.set_label(r'$\mathrm{p}_\mathrm{x}(x)$')

        return plt_data


class SimplexDiscrete(Simplex):
    def __init__(self, n, shape):
        self.n = n
        super().__init__(shape)

    def _set_x_plot(self):
        if self.shape in ((2,), (3,)):
            self.x_plt = simplex_grid(self.n, self._shape)
        else:
            self.x_plt = None
