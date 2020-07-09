import copy
import operator
import numpy as np
import matplotlib.pyplot as plt
from util.generic import check_data_shape, check_set_shape, vectorize_x_func

# TODO: getter and setters?
# TODO: plot methods

#%% Support objs
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




#%% Function objs

class BaseFunc(object):
    # def __init__(self):
    #     pass

    def __call__(self, x):
        raise NotImplementedError

    # def __add__(self, other):
    #     raise NotImplementedError


class NumericRangeFunc(BaseFunc):
    def __init__(self):
        super().__init__()
        self._mean = None

    def __call__(self, x):
        raise NotImplementedError

    @property
    def mean(self):
        return self._mean


# class DiscreteDomainFunc(BaseFunc):
#     pass


#%% IN USE

# def diag_func(f):
#     if isinstance(f, FiniteDomainFunc):
#         f_out = copy.deepcopy(f)
#         f_out.__call__ = lambda x_1, x_2: f


# TODO: scalar vs multi, numeric vs non


class FiniteDomainFunc(object):
    def __new__(cls, supp, val, set_shape=None):  # TODO: any object, not just numpy
        supp = np.asarray(supp)
        val = np.asarray(val)

        if np.issubdtype(val.dtype, np.number):
            if np.issubdtype(supp.dtype, np.number):
                return super().__new__(FiniteNumericDomainNumericFunc)
            else:
                return super().__new__(FiniteDomainNumericFunc)
        else:
            return super().__new__(cls)

    def __init__(self, supp, val, set_shape=None):
        if set_shape is None:
            set_shape = np.array(val).shape
        self._set_shape = set_shape
        self._set_size = int(np.prod(set_shape))

        self._supp, self.data_shape_x = check_set_shape(supp, set_shape)
        self._val, self.data_shape_y = check_set_shape(val, set_shape)

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
        self._val, self.data_shape_y = check_set_shape(val, self._set_shape)
        self._update_attr()

    def _update_attr(self):
        self._val_flat = self._val.reshape(self._set_size, -1)

    def _f(self, x):
        x_flat = np.asarray(x).flatten()
        _out = self._val_flat[(x_flat == self._supp_flat).all(-1)].reshape(self.data_shape_y)
        _out = _out.tolist()
        return _out     # TODO: exception if not in support

    def __call__(self, x):
        return vectorize_x_func(self._f, self.data_shape_x)(x)


class FiniteDomainNumericFunc(FiniteDomainFunc):

    def __init__(self, supp, val, set_shape=None):
        super().__init__(supp, val, set_shape)

    def _op_checker(self, other, op):
        if isinstance(other, FiniteDomainNumericFunc):
            if (self.supp == other.supp).all():
                return FiniteDomainNumericFunc(self.supp, op(self.val, other.val), self.set_shape)
        elif type(other) == float:
            return FiniteDomainNumericFunc(self.supp, op(self.val, other), self.set_shape)

    def __add__(self, other):
        return self._op_checker(other, operator.add)

    def __sub__(self, other):
        return self._op_checker(other, operator.sub)

    def __mul__(self, other):
        return self._op_checker(other, operator.mul)

    # def __rmul__(self, other):
    #     return self._op_checker(other, operator.mul)

    def __truediv__(self, other):
        return self._op_checker(other, operator.truediv)

    def __floordiv__(self, other):
        return self._op_checker(other, operator.floordiv)

    def __mod__(self, other):
        return self._op_checker(other, operator.mod)

    def __pow__(self, other):
        return self._op_checker(other, operator.pow)

    @property
    def max(self):
        return self._max

    @property
    def argmax(self):
        return self._argmax

    @property
    def min(self):
        return self._min

    @property
    def argmin(self):
        return self._argmin

    @property
    def sum(self):
        return self._sum

    def _update_attr(self):
        super()._update_attr()

        self._max = self._val_flat.max(0).reshape(self.data_shape_y)

        _temp = self._supp_flat[self._val_flat.argmax(0)].transpose()
        self._argmax = _temp.reshape(self.data_shape_x + self.data_shape_y)

        self._min = self._val_flat.min(0).reshape(self.data_shape_y)

        _temp = self._supp_flat[self._val_flat.argmin(0)].transpose()
        self._argmin = _temp.reshape(self.data_shape_x + self.data_shape_y)

        self._sum = self.val.sum()

    def plot(self, ax=None):
        if self.data_shape_y != ():
            raise ValueError("Can only plot scalar-valued functions.")

        if len(self.set_shape) == 1 and self.data_shape_x == ():
            if ax is None:
                _, ax = plt.subplots()
                ax.set(xlabel='$x$', ylabel='$y$')

            plt_data = ax.stem(self._supp, self._val, use_line_collection=True)

            return plt_data
        else:
            raise NotImplementedError('Plot method only implemented for 1-dimensional data.')


class FiniteNumericDomainNumericFunc(FiniteDomainNumericFunc):
    def __init__(self, supp, val, set_shape=None):
        super().__init__(supp, val, set_shape)

    @property
    def m1(self):
        return self._m1

    @property
    def m2c(self):
        return self._m2c

    def _update_attr(self):
        super()._update_attr()

        _mean_flat = (self._supp_flat[..., np.newaxis] * self._val_flat[:, np.newaxis]).sum(0)
        self._m1 = _mean_flat.reshape(self.data_shape_x + self.data_shape_y)

        ctr_flat = self._supp_flat[..., np.newaxis] - _mean_flat[np.newaxis]
        outer_flat = (ctr_flat[:, np.newaxis] * ctr_flat[:, :, np.newaxis])
        _temp = (outer_flat * self._val_flat[:, np.newaxis, np.newaxis]).sum(axis=0)
        self._m2c = _temp.reshape(2 * self.data_shape_x + self.data_shape_y)

    def plot(self, ax=None):
        if self.data_shape_y != ():
            raise ValueError("Can only plot scalar-valued functions.")

        set_ndim = len(self.set_shape)

        if set_ndim == 1 and self.data_shape_x == ():
            super().plot(self, ax)

        elif set_ndim in [2, 3]:
            if set_ndim == 2 and self.data_shape_x == (2,):
                if ax is None:
                    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
                    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$y$')

                plt_data = ax.bar3d(self._supp[..., 0].flatten(),
                                    self._supp[..., 1].flatten(), 0, 1, 1, self._val_flat.flatten(), shade=True)

            elif set_ndim == 3 and self.data_shape_x == (3,):
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
# val = np.random.random((3,4))
#
# a = FiniteDomainFunc(supp_x, val, set_shape=(3,))
# a.m1
# a.m2c
#
# supp_x = [[0,1], [1,1], [2,1]]
# set_shape = (3,)
# # val = [1,2,3]
# val = [FiniteDomainFunc(['a','b'], [8+i,9+i], (2,)) for i in range(3)]
# b = FiniteDomainFunc(supp_x, val, set_shape)
# b._f([2,1])
# b([2,1])
# b([[2,1],[0,1]])
#
# b.val
# (b+1).val
# (b-1).val
