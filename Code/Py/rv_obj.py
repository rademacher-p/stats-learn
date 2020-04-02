import numpy as np
from scipy import stats


#%% Deterministic RV, univariate
class DeterministicUniGen(stats.rv_continuous):

    def _cdf(self, x, *args):
        return np.where(x < 0, 0., 1.)

    def _pdf(self, x, *args):
        return np.where(x != 0, 0., np.inf)

    def _stats(self, *args, **kwds):
        return 0., 0., 0., 0.

    def _rvs(self, *args):
        return np.zeros(self._size)


deterministic_uni = DeterministicUniGen(name='deterministic')   # TODO: block non-singular inputs?


# class DeterministicUni(stats.rv_continuous):
#     def __init__(self, val=0):
#         super().__init__()
#         self.val = val
#         if not isinstance(self.val, (int, float)):
#             raise TypeError('Support must be a singular value.')
#
#     # def _argcheck(self, *args):
#     #     if not isinstance(self.val, (int, float)):
#     #         raise TypeError
#
#     def _cdf(self, x, *args):
#         return np.where(x < self.val, 0., 1.)
#
#     def _pdf(self, x, *args):
#         return np.where(x != self.val, 0., np.inf)
#
#     def _stats(self, *args, **kwds):
#         return self.val, 0., 0., 0.
#
#     def _rvs(self, *args):
#         return self.val * np.ones(self._size)


#%% Deterministic RV, multivariate

def _deterministic_check_parameters(val):
    return np.asarray(val)


def _deterministic_check_input(val, x):
    x = np.asarray(x)
    if x.shape != val.shape:
        raise TypeError(f'Input shape must be equal to {val.shape}.')
    return x


class DeterministicMultiGen(stats._multivariate.multi_rv_generic):

    # def __init__(self, seed=None):
    #     super(DeterministicMultiGen, self).__init__(seed)
    #     # self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)

    def __call__(self, val):
        return DeterministicMultiFrozen(val)

    def pdf(self, x, val):
        val = _deterministic_check_parameters(val)
        x = _deterministic_check_input(val, x)
        return np.where(x != 0, 0., np.inf)

    def mean(self, val):
        val = _deterministic_check_parameters(val)
        return val

    def var(self, val):
        val = _deterministic_check_parameters(val)
        return np.zeros(2*val.shape)

    def rvs(self, val, size=None):
        val = _deterministic_check_parameters(val)
        # if size is None:
        #     return val
        # elseif: type(size) is int:
        #     return np.tile(val[np.newaxis], ())


class DeterministicMultiFrozen(stats._multivariate.multi_rv_frozen):
    def __init__(self, val):
        self.val = _deterministic_check_parameters(val)
        self._dist = DeterministicMultiGen()

    def pdf(self, x):
        return self._dist.pdf(x, self.val)

    def mean(self):
        return self._dist.mean(self.val)

    def var(self):
        return self._dist.var(self.val)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.val, size, random_state)
