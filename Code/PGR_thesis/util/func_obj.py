import numpy as np
import matplotlib.pyplot as plt
from util.generic import check_data_shape, check_set_shape, vectorize_x_func

# TODO: getter and setters?
# TODO: plot methods


class BaseSupp(object):
    pass


class EuclideanSupp(BaseSupp):
    def __init__(self, lims):
        self.lims = np.asarray(lims)
        if self.lims.ndim == 1:
            self.lims = self.lims[np.newaxis]
        elif self.lims.ndim != 2:
            raise ValueError("Lims must be 2-D array.")
        if self.lims.shape[1] != 2:
            raise ValueError
        if not (self.lims[:, 0] <= self.lims[:, 1]).all():
            raise ValueError("Upper values must meet or exceed lower values:.")

    @property
    def ndim(self):
        return len(self.lims)


class FiniteSupp(BaseSupp):
    def __init__(self, vals):
        self.vals = np.asaray(vals)     # TODO: multidim?





class BaseFunc(object):
    # def __init__(self):
    #     pass

    def __call__(self, x):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError


class NumericRangeFunc(BaseFunc):
    def __init__(self):
        super().__init__()
        self._mean = None

    def __call__(self, x):
        raise NotImplementedError

    @property
    def mean(self):
        return self._mean


class DiscreteDomainFunc(BaseFunc):
    pass



class FiniteDomainFunc(object):

    def __new__(cls, supp, val, set_shape=()):
        supp = np.asarray(supp)
        val = np.asarray(val)
        if np.issubdtype(supp.dtype, np.number) and np.issubdtype(val.dtype, np.number):
            return super().__new__(FiniteDomainNumericFunc)
        else:
            return super().__new__(cls)

    def __init__(self, supp, val, set_shape=()):

        self._set_shape = set_shape
        self._set_size = int(np.prod(set_shape))
        self._supp, self._data_shape_x = check_set_shape(supp, set_shape)

        self._val, self._data_shape_y = check_set_shape(val, set_shape)

        self._supp_flat = self.supp.reshape(self._set_size, -1)
        if len(np.unique(self._supp_flat, axis=0)) < self._set_size:
            raise ValueError("Support elements must be unique.")

        self._update_attr()

    @property
    def set_shape(self):
        return self._set_shape

    @property
    def supp(self):
        return self._supp

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val, self._data_shape_y = check_set_shape(val, self._set_shape)
        self._update_attr()

    @property
    def mode(self):
        return self._mode

    def _update_attr(self):
        self._val_flat = self._val.reshape(self._set_size, -1)

        _temp = self._supp_flat[self._val_flat.argmax(0)].transpose()
        self._mode = _temp.reshape(self._data_shape_x + self._data_shape_y)

    def _f(self, x):
        x_flat = np.asarray(x).flatten()
        _out = self._val_flat[(x_flat == self._supp_flat).all(-1)].reshape(self._data_shape_y)
        return _out  # TODO: exception if not in support

    def __call__(self, x):
        return vectorize_x_func(self._f, self._data_shape_x)(x)

    def plot(self, ax=None):
        if self._data_shape_y != ():
            raise ValueError("Can only plot scalar-valued functions.")

        if len(self.set_shape) == 1 and self._data_shape_x == ():
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x$', ylabel='$y$')

            plt_data = ax.stem(self._supp, self._val, use_line_collection=True)

            return plt_data
        else:
            raise NotImplementedError('Plot method only implemented for 1-dimensional data.')


class FiniteDomainNumericFunc(FiniteDomainFunc):

    def __init__(self, supp, val, set_shape=()):
        super().__init__(supp, val, set_shape)

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    def _update_attr(self):
        super()._update_attr()

        _mean_flat = (self._supp_flat[..., np.newaxis] * self._val_flat[:, np.newaxis]).sum(0)
        self._mean = _mean_flat.reshape(self._data_shape_x + self._data_shape_y)

        ctr_flat = self._supp_flat[..., np.newaxis] - _mean_flat[np.newaxis]
        outer_flat = (ctr_flat[:, np.newaxis] * ctr_flat[:, :, np.newaxis])
        _temp = (outer_flat * self._val_flat[:, np.newaxis, np.newaxis]).sum(axis=0)
        self._cov = _temp.reshape(2 * self._data_shape_x + self._data_shape_y)

    def plot(self, ax=None):
        if self._data_shape_y != ():
            raise ValueError("Can only plot scalar-valued functions.")

        set_ndim = len(self.set_shape)

        if set_ndim == 1 and self._data_shape_x == ():
            super().plot(self, ax)

        elif set_ndim in [2, 3]:
            if set_ndim == 2 and self._data_shape_x == (2,):
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$y$')

                plt_data = ax.bar3d(self._supp[..., 0].flatten(),
                                    self._supp[..., 1].flatten(), 0, 1, 1, self._val_flat.flatten(), shade=True)

            elif set_ndim == 3 and self._data_shape_x == (3,):
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$')

                plt_data = ax.scatter(self._supp[..., 0], self._supp[..., 1], self._supp[..., 2], s=15, c=self._val)

                c_bar = plt.colorbar(plt_data)
                c_bar.set_label('$y$')

            else:
                raise ValueError

            return plt_data

        else:
            raise NotImplementedError('Plot method only implemented for 1- and 2- dimensional data.')


# #
# supp_x = np.stack(np.meshgrid(np.arange(2), np.arange(3)), axis=-1)
# val = np.random.random((3,))
#
# a = FiniteDomainFunc(supp_x, val, set_shape=(3,))
# a.mean
# a.cov


