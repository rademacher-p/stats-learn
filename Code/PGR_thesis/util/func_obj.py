import itertools
import numpy as np
# from scipy.special import binom


class FiniteFunc:
    def __init__(self, domain, codomain):
        if len(domain) != len(codomain):
            raise ValueError("Domain and co-domain must have equal sizes.")

        self.domain = np.asarray(domain)
        self.codomain = np.asarray(codomain)

        self.size = len(self.domain)
        self.domain_shape = self.domain.shape[1:]
        self.codomain_shape = self.codomain.shape[1:]

        # self.func = np.asarray(list(zip(domain, codomain)),
        #                        dtype=[('domain', domain.dtype, self.domain_shape),
        #                               ('codomain', codomain.dtype, self.codomain_shape)])

        self._domain_flat = self.domain.reshape(self.size, -1)
        if len(np.unique(self._domain_flat, axis=0)) < self.size:
            raise ValueError("Domain elements must be unique.")

    def __call__(self, x):
        x = np.asarray(x)
        _out = self.domain[x.flatten() == self._domain_flat]
        if _out.shape[0] == 1:
            return _out.squeeze(axis=0)
        else:
            raise ValueError("Input is not in the function domain.")