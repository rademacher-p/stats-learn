import numpy as np
from util.generic import check_data_shape, check_set_shape, vectorize_x_func

# TODO: getter and setters?

class BaseFunc(object):
    # def __init__(self):
    #     pass

    def __call__(self, x):
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



class FiniteDomainFunc(DiscreteDomainFunc):
    def __init__(self, supp, f, set_shape=()):

        self.set_shape = set_shape
        self.set_size = int(np.prod(set_shape))
        self.supp, self.data_shape_x = check_set_shape(supp, set_shape)

        self.f = f

        self._supp_flat = self.supp.reshape(self.set_size, -1)
        if len(np.unique(self._supp_flat, axis=0)) < self.set_size:
            raise ValueError("Support elements must be unique.")

        self._mode = None
        self._mean = None
        self.update_attr()

    def __call__(self, x):
        return vectorize_x_func(self.f, self.data_shape_x)(x)

    @property
    def mode(self):
        return self._mode

    @property
    def mean(self):
        return self._mean

    def update_attr(self):
        vals, data_shape_y = check_set_shape(self(self.supp), self.set_shape)
        vals_flat = vals.reshape(self.set_size, -1)

        self._mode = self._supp_flat[vals_flat.argmax(0)].reshape(data_shape_y + self.data_shape_x)     # TODO: dim order?


        _temp = (self._supp_flat[..., np.newaxis] * vals_flat[:, np.newaxis]).sum(0)
        self._mean = _temp.reshape(self.data_shape_x + data_shape_y)

    @classmethod
    def gen_explicit(cls, supp, val, set_shape=()):

        val, data_shape_y = check_set_shape(val, set_shape)
        set_size = int(np.prod(set_shape))
        val_flat = val.reshape(set_size, -1)

        supp_flat = supp.reshape(set_size, -1)

        def f(x):
            x_flat = np.asarray(x).flatten()
            _out = val_flat[(x_flat == supp_flat).all(-1)].reshape(data_shape_y)
            return _out
            # if _out.shape[0] == 1:
            #     return _out.squeeze(0)
            # else:
            #     raise ValueError("Input is not in the function supp.")

        return cls(supp, f, set_shape=set_shape)


# #
# supp_x = np.stack(np.meshgrid(np.arange(2), np.arange(3)), axis=-1)
# val = np.random.random((3, 2))
#
# a = FiniteDomainFunc.gen_explicit(supp_x, val, set_shape=(3,2))
# a(a.supp[1, 1])
# a([a.supp[1, 1], a.supp[0, 1]])
# a.mean