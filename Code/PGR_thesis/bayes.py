"""
Bayesian Prior objects.
"""

import types
import functools
# import itertools
import numpy as np
# from scipy.stats._multivariate import multi_rv_generic

from RE_obj import FiniteRE, DirichletRV
from SL_obj import YcXModel


#%% Priors

# TODO: COMPLETE property set/get check, rework!


class BaseBayes:
    def __init__(self, model_gen, model_kwargs=None, prior=None):
        # super().__init__(rng)

        if model_kwargs is None:
            model_kwargs = {}

        self.model_gen = functools.partial(model_gen, **model_kwargs)
        self.prior = prior

    def random_model(self):     # TODO: defaults to deterministic prior!?
        return self.model_gen()


class FiniteREBayes(BaseBayes):
    def __init__(self, supp_x, supp_y, prior, rng_model=None):     # TODO: rng check, None default?
        self.supp_x = supp_x
        self.supp_y = supp_y        # Assumed to be my SL structured array!

        self._supp_shape_x = supp_x.shape
        self._supp_shape_y = supp_y.shape
        self._data_shape_x = supp_x.dtype['x'].shape
        self._data_shape_y = supp_y.dtype['y'].shape

        model_gen = YcXModel.finite_model
        model_kwargs = {'supp_x': supp_x['x'], 'supp_y': supp_y['y'], 'rng': rng_model}
        super().__init__(model_gen, model_kwargs, prior)

    def random_model(self):
        raise NotImplementedError("Method must be overwritten.")


class FiniteDirichletBayes(FiniteREBayes):

    # TODO: initialization from marginal/conditional? full mean init as classmethod?
    def __init__(self, supp_x, supp_y, alpha_0, mean, rng_model=None, rng_prior=None):
        prior = DirichletRV(alpha_0, mean, rng_prior)
        super().__init__(supp_x, supp_y, prior, rng_model)

    def random_model(self, rng=None):
        p = self.prior.rvs(random_state=rng)        # TODO: generate using marginal/conditional independence??

        p_x = p.reshape(self._supp_shape_x + (-1,)).sum(axis=-1)

        def p_y_x(x):
            _p_flat = p.reshape((-1,) + self._supp_shape_y)
            _p_slice = _p_flat[np.all(x.flatten()
                                      == self.supp_x['x'].reshape(self.supp_x.size, -1), axis=-1)].squeeze(axis=0)
            p_y = _p_slice / _p_slice.sum()
            return p_y

        return self.model_gen(p_x=p_x, p_y_x=p_y_x)








# ###
# class BayesRE:
#     def __init__(self, prior, rand_kwargs, model_gen, model_kwargs=None):
#         if model_kwargs is None:
#             model_kwargs = {}
#         self.model_kwargs = model_kwargs
#
#         # prior.rand_kwargs = types.MethodType(rand_kwargs, prior)
#         self.prior = prior
#         self.rand_kwargs = rand_kwargs
#
#         self.model_gen = model_gen
#
#     def random_model(self):
#         return self.model_gen(**self.model_kwargs, **self.rand_kwargs(self.prior))
#
#     @classmethod
#     def finite_dirichlet(cls, supp_x, supp_y, alpha_0, mean, rng):
#         prior = DirichletRV(alpha_0, mean)
#
#         def rand_kwargs(dist):
#             p = dist.rvs()
#
#             p_x = p.reshape(supp_x.shape + (-1,)).sum(axis=-1)
#
#             def p_y_x(x):
#                 _temp = p.reshape((-1,) + supp_y.shape)[np.all(x.flatten() == supp_x['x'].reshape(supp_x.size, -1),
#                                                                axis=-1)].squeeze(axis=0)
#                 p_y = _temp / _temp.sum()
#                 return p_y
#
#             return {'p_x': p_x, 'p_y_x': p_y_x}
#
#         model_gen = YcXModel.finite_model
#         model_kwargs = {'supp_x': supp_x['x'], 'supp_y': supp_y['y'], 'rng': rng}
#
#         return cls(prior, rand_kwargs, model_gen, model_kwargs)










# def bayes_re(model_gen, prior, rand_kwargs, model_kwargs=None):
#     if model_kwargs is None:
#         model_kwargs = {}
#
#     prior.rand_kwargs = types.MethodType(rand_kwargs, prior)
#
#     obj = model_gen(**model_kwargs, **prior.rand_kwargs())
#     obj.prior = prior
#
#     def random_model(self):
#         for attr, val in prior.rand_kwargs().items():
#             setattr(self, attr, val)
#
#     obj.random_model = types.MethodType(random_model, obj)
#
#     return obj
#
#
# def finite_bayes(set_x_s, set_y_s, alpha_0, mean, rng):
#
#     model_gen = YcXModel.finite_model
#     model_kwargs = {'set_x': set_x_s['x'], 'set_y': set_y_s['y'], 'rng': rng}
#
#     prior = DirichletRV(alpha_0, mean)
#
#
#     def rand_kwargs(self):
#         p = self.rvs()
#
#         p_x = p.reshape(set_x_s.shape + (-1,)).sum(axis=-1)
#
#         def p_y_x(x):
#             _temp = p.reshape((-1,) + set_y_s.shape)[np.all(x.flatten() == set_x_s['x'].reshape(set_x_s.size, -1),
#                                                             axis=-1)].squeeze(axis=0)
#             p_y = _temp / _temp.sum()
#             return p_y
#
#         return {'p_x': p_x, 'p_y_x': p_y_x}
#
#     return bayes_re(model_gen, prior, rand_kwargs, model_kwargs)
#
#
#
#
# def bayes_re2(model_gen, prior):
#
#     obj = model_gen(**prior.rand_kwargs())
#     obj.prior = prior
#
#     def random_model(self):
#         for attr, val in prior.rand_kwargs().items():
#             setattr(self, attr, val)
#
#     obj.random_model = types.MethodType(random_model, obj)
#
#     return obj
#
#
# def finite_bayes2(set_x_s, set_y_s, alpha_0, mean, rng):
#
#     model_gen = functools.partial(YcXModel.finite_model,
#                                   **{'set_x': set_x_s['x'], 'set_y': set_y_s['y'], 'rng': rng})
#
#     def rand_kwargs(self):
#         p = self.rvs()
#
#         p_x = p.reshape(set_x_s.shape + (-1,)).sum(axis=-1)
#
#         def p_y_x(x):
#             _temp = p.reshape((-1,) + set_y_s.shape)[np.all(x.flatten() == set_x_s['x'].reshape(set_x_s.size, -1),
#                                                             axis=-1)].squeeze(axis=0)
#             p_y = _temp / _temp.sum()
#             return p_y
#
#         return {'p_x': p_x, 'p_y_x': p_y_x}
#
#     prior = DirichletRV(alpha_0, mean)
#     prior.rand_kwargs = types.MethodType(rand_kwargs, prior)
#
#     return bayes_re2(model_gen, prior)








#%% Boneyard

# class BayesRE:
#     def __new__(cls, prior, model_cls, model_kwargs):
#         self.prior = prior
#
#         class cls_frozen(model_cls):
#             __new__ = functools.partialmethod(model_cls.__new__, **model_kwargs)
#             __init__ = functools.partialmethod(model_cls.__init__, **model_kwargs)
#
#         self.cls_frozen = cls_frozen
#
#     def random_model(self):
#         args = self.dist.rvs()
#         return self.cls_frozen(p=args)

# class FiniteREPrior(FiniteRE):
#     def __new__(cls, supp, seed=None):
#         cls.prior = DirichletRV(2, [.5, .5])
#         p = cls.rng_prior()
#         return super().__new__(cls, supp, p, seed)
#
#     def __init__(self, supp, seed=None):
#         # self.prior = DirichletRV(2, [.5, .5])
#         p = self.rng_prior()
#         super().__init__(supp, p, seed)
#
#     @classmethod
#     def rng_prior(cls):
#         return cls.prior.rvs()
#
#     def random_model(self):
#         self.p = self.rng_prior()
#
# f = FiniteREPrior(['a','b'])



# class Prior:
#     def __init__(self, dist, model_cls):
#         self.dist = dist
#         self.model_cls = model_cls
#
#         args = self.dist.rvs()
#         self.model = self.model_cls(p=args)
#
#     # def random_model(self):
#     #     args = self.dist.rvs()
#     #     return self.model_cls(p=args)
#
#     def randomize(self):
#         args = self.dist.rvs()
#         self.model.p = args
#
#
# dist = DirichletRV(4, [.5, .5])
# class model_cls(FiniteRE):
#     __new__ = functools.partialmethod(FiniteRE.__new__, supp=['a', 'c'])
#     __init__ = functools.partialmethod(FiniteRE.__init__, supp=['a', 'c'])
#
# # a = model_cls(p=[.4,.6])
#
# d = Prior(dist, model_cls)
# # d.random_model()
#
#
#
# class Prior2:
#     def __init__(self, dist, model_cls, model_kwargs):
#         self.dist = dist
#
#         class cls_frozen(model_cls):
#             __new__ = functools.partialmethod(model_cls.__new__, **model_kwargs)
#             __init__ = functools.partialmethod(model_cls.__init__, **model_kwargs)
#
#         self.cls_frozen = cls_frozen
#
#     def random_model(self):
#         args = self.dist.rvs()
#         return self.cls_frozen(p=args)
#
#
# dist = DirichletRV(4, [.5, .5])
# model_cls = FiniteRE
# model_kwargs = {'supp': ['d', 'e']}
#
# d = Prior2(dist, model_cls, model_kwargs)
# d.random_model()
#
#
#
# class BayesRE:
#     def __init__(self, prior, model):
#         self.prior = prior
#         self.model = model
#
#     def random_model(self):
#         self.prior.rvs()
#         return None







# class FiniteREPrior(Prior):
#     def __init__(self, dist, support, seed=None):
#         super().__init__(dist, seed)
#         self.support = support
#         # self.support_y = np.unique(support['y'])
#         # self.support_x = np.unique(support['x'])
#
#     def random_model(self):
#         theta_pmf = self.dist.rvs()
#         # theta_pmf_m = None
#
#         theta = FiniteRE(self.support, theta_pmf, self.random_state)
#         # theta_m = FiniteRE(self.support_x, theta_pmf_m, self.random_state)
#
#         return theta
#
#     @classmethod
#     def dirichlet_prior(cls, alpha_0, mean, support, seed=None):
#         return cls(DirichletRV(alpha_0, mean), support, seed)
#
#
#
# class DatPriorDoe(Prior):
#     def __init__(self, loc, scale):
#         dist = stats.rayleigh(loc, scale)
#         super().__init__(dist, seed)
#
#     def random_model(self):
#         a, b = self.dist.rvs(size=2)
#         theta_m = stats.beta(a, b)
#         def theta_c(x): return stats.beta(5*x, 5*(1-x))
#
#         return YcXModel(theta_m, theta_c)
