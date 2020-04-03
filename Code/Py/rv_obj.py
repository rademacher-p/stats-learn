import numpy as np
from scipy import stats
import warnings


#%% Deterministic RV, univariate

# def _deterministic_uni_check_parameters(val):
#     val = np.asarray(val).squeeze()
#     if val.size != 1:
#         raise TypeError('Parameter must be singular.')
#     return val
#
#
# def _deterministic_uni_check_input(val, x):
#     val = _deterministic_uni_check_parameters(val)
#     x = np.asarray(x).squeeze()
#     if x.shape != val.shape:
#         raise TypeError(f'Input must be singular.')
#     return x


class DeterministicUniGen(stats.rv_continuous):

    def _cdf(self, x, *args):
        return np.where(x < 0, 0., 1.)

    def _pdf(self, x, *args):
        return np.where(x != 0, 0., np.inf)

    def _stats(self, *args, **kwds):
        return 0., 0., 0., 0.

    def _rvs(self, *args):
        return np.zeros(self._size)

    def median(self, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        return float(loc)

    def mode(self, *args, **kwds):      # TODO: cannot be accessed through Frozen RV, no method for rv_frozen
        args, loc, scale = self._parse_args(*args, **kwds)
        # loc, scale = map(np.asarray, (loc, scale))
        return float(loc)


deterministic_uni = DeterministicUniGen(name='deterministic')   # TODO: block non-singular inputs?


#%% Deterministic RV, multivariate

def _deterministic_multi_check_parameters(val):
    return np.asarray(val)
    # val = np.asarray(val)
    # if 1 in val.shape:
    #     warnings.warn('\nSingleton dimensions have been squeezed out of input parameter.')
    # return val.squeeze()


def _deterministic_multi_check_input(val, x):
    # val = _deterministic_multi_check_parameters(val)      # already performed by calling method
    x = np.asarray(x)
    if not (x.shape == val.shape or (len(x.shape) == len(val.shape) + 1 and x.shape[1:] == val.shape)):
        raise TypeError(f'\nInput shape must be equal to {val.shape} or (i,)+{val.shape}.')
    return x

    # if x.shape == val.shape:
    #     return x[np.newaxis]    # reshape to common format
    # elif len(x.shape) == len(val.shape) + 1 and x.shape[1:] == val.shape:
    #     return x
    # else:
    #     raise TypeError(f'\nInput shape must be equal to {val.shape} or (i,)+{val.shape}.')


class DeterministicMultiGen(stats._multivariate.multi_rv_generic):

    # TODO: docstring?
    # def __init__(self, seed=None):
    #     super(DeterministicMultiGen, self).__init__(seed)
    #     # self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)

    # TODO: lots of redundant parameter check calls, even when frozen.

    def __call__(self, val):
        val = _deterministic_multi_check_parameters(val)
        return DeterministicMultiFrozen(val)

    def cdf(self, x, val):
        val = _deterministic_multi_check_parameters(val)
        x = _deterministic_multi_check_input(val, x)
        return np.where(np.all(x >= val, axis=tuple(range(1, x.ndim))), 1., 0.).squeeze()

    # def cdf_single(self, x, val):
    #     return 1. if np.all(x >= val) else 0.
    #
    # def cdf(self, x, val):      # TODO: move these general broadcasting methods to a superclass
    #     val = _deterministic_multi_check_parameters(val)
    #     x = _deterministic_multi_check_input(val, x)
    #     return np.vectorize(self.cdf_single, signature='(n,...,m),(n,...,m)->()')(x, val)   #TODO: BROKEN SIGS!

    def pdf(self, x, val):
        val = _deterministic_multi_check_parameters(val)
        x = _deterministic_multi_check_input(val, x)
        return np.where(np.all(x == val, axis=tuple(range(1, x.ndim))), np.inf, 0.).squeeze()

    # def pdf_single(self, x, val):
    #     return np.inf if np.all(x == val) else 0.
    #
    # def pdf(self, x, val):        # TODO: move these general broadcasting methods to a superclass
    #     val = _deterministic_multi_check_parameters(val)
    #     x = _deterministic_multi_check_input(val, x)
    #     return np.vectorize(self.pdf_single, signature='(m,n),(m,n)->()')(x, val)   # TODO: BROKEN SIGS!

    def mean(self, val):
        val = _deterministic_multi_check_parameters(val)
        return val

    def var(self, val):
        val = _deterministic_multi_check_parameters(val)
        return np.zeros(2*val.shape)

    def median(self, val):
        val = _deterministic_multi_check_parameters(val)
        return val

    def mode(self, val):
        val = _deterministic_multi_check_parameters(val)
        return val

    def rvs(self, val, size=1):
        val = _deterministic_multi_check_parameters(val)
        if size == 1:
            return val
        else:
            return np.broadcast_to(val[np.newaxis], (size,)+val.shape)


deterministic_multi = DeterministicMultiGen()


class DeterministicMultiFrozen(stats._multivariate.multi_rv_frozen):
    def __init__(self, val):
        self.val = val
        self._dist = DeterministicMultiGen()

    def cdf(self, x):
        return self._dist.cdf(x, self.val)

    def pdf(self, x):
        return self._dist.pdf(x, self.val)

    def mean(self):
        return self._dist.mean(self.val)

    def var(self):
        return self._dist.var(self.val)

    def median(self):
        return self._dist.median(self.val)

    def mode(self):
        return self._dist.mode(self.val)

    def rvs(self, size=1):
        return self._dist.rvs(self.val, size)
