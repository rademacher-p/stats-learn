"""
Bayesian Prior objects.
"""

import types
import functools
# import itertools
import numpy as np
# from scipy.stats._multivariate import multi_rv_generic

import RE_obj
import RE_obj_callable
from SL_obj import YcXModel
from util.generic import empirical_pmf

from util.func_obj import FiniteDomainFunc

#%% Priors

# TODO: Add deterministic DEP to effect a DP realization and sample!!
# TODO: COMPLETE property set/get check, rework!


class BaseBayes:
    def __init__(self, model_gen, model_kwargs=None, prior=None):
        # super().__init__(rng)

        if model_kwargs is None:
            model_kwargs = {}

        self.model_gen = functools.partial(model_gen, **model_kwargs)
        self.prior = prior

    def random_model(self):     # defaults to deterministic bayes_model!?
        return self.model_gen()

    def posterior_mean(self, d):  # TODO: generalize method for base classes, full posterior object?
        raise NotImplementedError


# class BetaModelBayes(BaseBayes):
#     def __init__(self, prior=None, rng_model=None):
#         model_gen = YcXModel.beta_model
#         model_kwargs = {'a': .9, 'b': .9, 'c': 5, 'rng': rng_model}
#         super().__init__(model_gen, model_kwargs, prior)
#
#     # def random_model(self):
#     #     raise NotImplementedError("Method must be overwritten.")


# class NormalModelBayes(BaseBayes):
#     def __init__(self, prior=None, rng_model=None):
#         model_gen = YcXModel.norm_model
#         model_kwargs = {'mean_x': 0, 'var_x': 1, 'var_y': 1, 'rng': rng_model}
#         super().__init__(model_gen, model_kwargs, prior)
#
#     # def random_model(self):
#     #     raise NotImplementedError("Method must be overwritten.")


class DirichletFiniteYcXModelBayesNew(BaseBayes):

    def __init__(self, alpha_0, mean_x, mean_y_x, rng_model=None, rng_prior=None):
        model_gen = YcXModel.finite_model
        model_kwargs = {'rng': rng_model}
        prior = {'p_x': RE_obj_callable.DirichletRV(alpha_0, mean_x, rng_prior),
                 'p_y_x': lambda x: RE_obj_callable.DirichletRV(alpha_0 * mean_x(x), mean_y_x(x), rng_prior)}
        super().__init__(model_gen, model_kwargs, prior)

        self.alpha_0 = alpha_0
        self._mean_x = mean_x
        self._mean_y_x = mean_y_x

        self._supp_x = mean_x.supp
        self._data_shape_x = mean_x.data_shape_x
        self._supp_y = mean_y_x.val.flatten()[0].supp
        self._data_shape_y = mean_y_x.val.flatten()[0].data_shape_x

    def random_model(self, rng=None):
        p_x = self.prior['p_x'].rvs(random_state=rng)

        val = []        # FIXME: better way?
        for x_flat in self._mean_x._supp_flat:
            x = x_flat.reshape(self._data_shape_x)
            val.append(self.prior['p_y_x'](x).rvs(random_state=rng))
        val = np.array(val).reshape(self._mean_x.set_shape)
        p_y_x = FiniteDomainFunc(self._supp_x, val)

        return self.model_gen(p_x=p_x, p_y_x=p_y_x)

    def posterior_mean(self, d):
        n = len(d)
        if n == 0:
            return self.model_gen(p_x=self._mean_x, p_y_x=self._mean_y_x)

        emp_dist_x = FiniteDomainFunc(self._supp_x,
                                      empirical_pmf(d['x'], self._supp_x, self._data_shape_x))

        c_prior_x = 1 / (1 + n / self.alpha_0)
        p_x = c_prior_x * self._mean_x + (1 - c_prior_x) * emp_dist_x

        def emp_dist_y_x(x):
            x = np.asarray(x)
            d_match = d[np.all(x.flatten() == d['x'].reshape(n, -1), axis=-1)].squeeze()
            return FiniteDomainFunc(self._supp_y,
                                    empirical_pmf(d_match['y'], self._supp_y, self._data_shape_y))

        def p_y_x(x):   # TODO: arithmetic of functionals?!?
            c_prior_y = 1 / (1 + (n * emp_dist_x(x)) / (self.alpha_0 * self._mean_x(x)))
            return c_prior_y * self._mean_y_x(x) + (1 - c_prior_y) * emp_dist_y_x(x)

        return self.model_gen(p_x=p_x, p_y_x=p_y_x)     # TODO: RE obj or just func obj?

    def predictive_dist(self, d):
        n = d.size
        if n == 0:
            return self._mean_y_x

        # emp_dist_x = FiniteDomainFunc(self._supp_x,
        #                               empirical_pmf(d['x'], self._supp_x, self._data_shape_x))
        #
        # c_prior_x = 1 / (1 + n / self.alpha_0)
        # p_x = c_prior_x * self._mean_x + (1 - c_prior_x) * emp_dist_x

        # def emp_dist_y_x(x):
        #     x = np.asarray(x)
        #     d_match = d[np.all(x.flatten() == d['x'].reshape(n, -1), axis=-1)].squeeze()
        #     return FiniteDomainFunc(self._supp_y,
        #                             empirical_pmf(d_match['y'], self._supp_y, self._data_shape_y))

        def p_y_x(x):   # TODO: arithmetic of functionals?!?
            x = np.asarray(x)
            d_match = d[np.all(x.flatten() == d['x'].reshape(n, -1), axis=-1)].squeeze()
            n_match = d_match.size
            if n_match == 0:
                return self._mean_y_x(x)

            c_prior_y = 1 / (1 + n_match / (self.alpha_0 * self._mean_x(x)))
            emp_dist = FiniteDomainFunc(self._supp_y, empirical_pmf(d_match['y'], self._supp_y, self._data_shape_y))
            return c_prior_y * self._mean_y_x(x) + (1 - c_prior_y) * emp_dist

        return p_y_x



#%% Without func objects
class FiniteYcXModelBayes(BaseBayes):
    def __init__(self, supp_x, supp_y, prior, rng_model=None):
        self.supp_x = supp_x
        self.supp_y = supp_y        # TODO: Assumed to be my SL structured array!

        self._supp_shape_x = supp_x.shape
        self._supp_shape_y = supp_y.shape
        self._data_shape_x = supp_x.dtype['x'].shape
        self._data_shape_y = supp_y.dtype['y'].shape

        model_gen = YcXModel.finite_model_orig
        # model_kwargs = {'rng': rng_model}
        model_kwargs = {'supp_x': supp_x['x'], 'supp_y': supp_y['y'], 'rng': rng_model}
        super().__init__(model_gen, model_kwargs, prior)

    def random_model(self):
        raise NotImplementedError("Method must be overwritten.")


class DirichletFiniteYcXModelBayes(FiniteYcXModelBayes):

    # TODO: initialization from marginal/conditional? full mean init as classmethod?

    def __init__(self, supp_x, supp_y, alpha_0, mean, rng_model=None, rng_prior=None):
        prior = RE_obj.DirichletRV(alpha_0, mean, rng_prior)
        super().__init__(supp_x, supp_y, prior, rng_model)

        self.alpha_0 = self.prior.alpha_0
        self.mean = self.prior.mean

        self._mean_x = self.mean.reshape(self._supp_shape_x + (-1,)).sum(axis=-1)

        def _mean_y_x(x):
            x = np.asarray(x)
            _mean_flat = self.mean.reshape((-1,) + self._supp_shape_y)
            _mean_slice = _mean_flat[np.all(x.flatten()
                                            == self.supp_x['x'].reshape(self.supp_x.size, -1), axis=-1)].squeeze(axis=0)
            mean_y = _mean_slice / _mean_slice.sum()
            return mean_y

        self._mean_y_x = _mean_y_x

        # self._model_gen = functools.partial(YcXModel.finite_model, supp_x=supp_x['x'], supp_y=supp_y['y'], rng=None)

    def random_model(self, rng=None):
        p = self.prior.rvs(random_state=rng)

        p_x = p.reshape(self._supp_shape_x + (-1,)).sum(axis=-1)

        def p_y_x(x):
            x = np.asarray(x)
            _p_flat = p.reshape((-1,) + self._supp_shape_y)
            _p_slice = _p_flat[np.all(x.flatten()
                                      == self.supp_x['x'].reshape(self.supp_x.size, -1), axis=-1)].squeeze(axis=0)
            p_y = _p_slice / _p_slice.sum()
            return p_y

        return self.model_gen(p_x=p_x, p_y_x=p_y_x)

    def posterior_mean(self, d):
        n = len(d)

        if n == 0:
            p_x, p_y_x = self._mean_x, self._mean_y_x
        else:
            emp_dist_x = empirical_pmf(d['x'], self.supp_x['x'], self._data_shape_x)

            def emp_dist_y_x(x):
                x = np.asarray(x)
                d_match = d[np.all(x.flatten() == d['x'].reshape(n, -1), axis=-1)].squeeze()
                return empirical_pmf(d_match['y'], self.supp_y['y'], self._data_shape_y)

            c_prior_x = 1 / (1 + n / self.alpha_0)
            p_x = c_prior_x * self._mean_x + (1 - c_prior_x) * emp_dist_x

            def p_y_x(x):
                x = np.asarray(x)
                i = (self.supp_x['x'].reshape(self._supp_shape_x + (-1,)) == x.flatten()).all(-1)
                c_prior_y = 1 / (1 + (n * emp_dist_x[i]) / (self.alpha_0 * self._mean_x[i]))
                return c_prior_y * self._mean_y_x(x) + (1 - c_prior_y) * emp_dist_y_x(x)

        return self.model_gen(p_x=p_x, p_y_x=p_y_x)





# #%%
# class FiniteYcXModelBayesOrig(BaseBayes):
#     def __init__(self, supp_x, supp_y, prior, rng_model=None):     # TODO: rng check, None default?
#         self.supp_x = supp_x
#         self.supp_y = supp_y        # TODO: Assumed to be my SL structured array!
#
#         self._supp_shape_x = supp_x.shape
#         self._supp_shape_y = supp_y.shape
#         self._data_shape_x = supp_x.dtype['x'].shape
#         self._data_shape_y = supp_y.dtype['y'].shape
#
#         model_gen = YcXModel.finite_model_orig
#         model_kwargs = {'supp_x': supp_x['x'], 'supp_y': supp_y['y'], 'rng': rng_model}
#         super().__init__(model_gen, model_kwargs, prior)
#
#     def random_model(self):
#         raise NotImplementedError("Method must be overwritten.")
#
#
# class DirichletFiniteYcXModelBayesOrig(FiniteYcXModelBayesOrig):
#
#     # TODO: initialization from marginal/conditional? full mean init as classmethod?
#
#     def __init__(self, supp_x, supp_y, alpha_0, mean, rng_model=None, rng_prior=None):
#         prior = DirichletRV(alpha_0, mean, rng_prior)
#         super().__init__(supp_x, supp_y, prior, rng_model)
#
#         self.alpha_0 = self.prior.alpha_0
#         self.mean = self.prior.mean
#
#         self._mean_x = self.mean.reshape(self._supp_shape_x + (-1,)).sum(axis=-1)
#
#         def _mean_y_x(x):
#             x = np.asarray(x)
#             _mean_flat = self.mean.reshape((-1,) + self._supp_shape_y)
#             _mean_slice = _mean_flat[np.all(x.flatten()
#                                             == self.supp_x['x'].reshape(self.supp_x.size, -1), axis=-1)].squeeze(axis=0)
#             mean_y = _mean_slice / _mean_slice.sum()
#             return mean_y
#
#         self._mean_y_x = _mean_y_x
#
#         self._model_gen = functools.partial(YcXModel.finite_model_orig,
#                                             supp_x=supp_x['x'], supp_y=supp_y['y'], rng=None)
#
#     def random_model(self, rng=None):
#         p = self.prior.rvs(random_state=rng)
#
#         p_x = p.reshape(self._supp_shape_x + (-1,)).sum(axis=-1)
#
#         def p_y_x(x):
#             x = np.asarray(x)
#             _p_flat = p.reshape((-1,) + self._supp_shape_y)
#             _p_slice = _p_flat[np.all(x.flatten()
#                                       == self.supp_x['x'].reshape(self.supp_x.size, -1), axis=-1)].squeeze(axis=0)
#             p_y = _p_slice / _p_slice.sum()
#             return p_y
#
#         return self.model_gen(p_x=p_x, p_y_x=p_y_x)
#
#     def posterior_mean(self, d):
#         n = len(d)
#
#         if n == 0:
#             p_x, p_y_x = self._mean_x, self._mean_y_x
#         else:
#             emp_dist_x = empirical_pmf(d['x'], self.supp_x['x'], self._data_shape_x)
#
#             def emp_dist_y_x(x):
#                 x = np.asarray(x)
#                 d_match = d[np.all(x.flatten() == d['x'].reshape(n, -1), axis=-1)].squeeze()
#                 if d_match.size == 0:
#                     return np.empty(self._supp_shape_y)
#                 return empirical_pmf(d_match['y'], self.supp_y['y'], self._data_shape_y)
#
#             c_prior_x = 1 / (1 + n / self.alpha_0)
#             p_x = c_prior_x * self._mean_x + (1 - c_prior_x) * emp_dist_x
#
#             def p_y_x(x):
#                 x = np.asarray(x)
#                 i = (self.supp_x['x'].reshape(self._supp_shape_x + (-1,)) == x.flatten()).all(-1)
#                 c_prior_y = 1 / (1 + (n * emp_dist_x[i]) / (self.alpha_0 * self._mean_x[i]))
#                 return c_prior_y * self._mean_y_x(x) + (1 - c_prior_y) * emp_dist_y_x(x)
#
#         return self._model_gen(p_x=p_x, p_y_x=p_y_x)






# ###
# class BayesRE:
#     def __init__(self, bayes_model, rand_kwargs, model_gen, model_kwargs=None):
#         if model_kwargs is None:
#             model_kwargs = {}
#         self.model_kwargs = model_kwargs
#
#         # bayes_model.rand_kwargs = types.MethodType(rand_kwargs, bayes_model)
#         self.bayes_model = bayes_model
#         self.rand_kwargs = rand_kwargs
#
#         self.model_gen = model_gen
#
#     def random_model(self):
#         return self.model_gen(**self.model_kwargs, **self.rand_kwargs(self.bayes_model))
#
#     @classmethod
#     def finite_dirichlet(cls, supp, supp_y, alpha_0, mean, rng):
#         bayes_model = DirichletRV(alpha_0, mean)
#
#         def rand_kwargs(dist):
#             p = dist.rvs()
#
#             p_x = p.reshape(supp.shape + (-1,)).sum(axis=-1)
#
#             def p_y_x(x):
#                 _temp = p.reshape((-1,) + supp_y.shape)[np.all(x.flatten() == supp['x'].reshape(supp.size, -1),
#                                                                axis=-1)].squeeze(axis=0)
#                 p_y = _temp / _temp.sum()
#                 return p_y
#
#             return {'p_x': p_x, 'p_y_x': p_y_x}
#
#         model_gen = YcXModel.finite_model
#         model_kwargs = {'supp': supp['x'], 'supp_y': supp_y['y'], 'rng': rng}
#
#         return cls(bayes_model, rand_kwargs, model_gen, model_kwargs)










# def bayes_re(model_gen, bayes_model, rand_kwargs, model_kwargs=None):
#     if model_kwargs is None:
#         model_kwargs = {}
#
#     bayes_model.rand_kwargs = types.MethodType(rand_kwargs, bayes_model)
#
#     obj = model_gen(**model_kwargs, **bayes_model.rand_kwargs())
#     obj.bayes_model = bayes_model
#
#     def random_model(self):
#         for attr, val in bayes_model.rand_kwargs().items():
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
#     model_kwargs = {'supp': set_x_s['x'], 'set_y': set_y_s['y'], 'rng': rng}
#
#     bayes_model = DirichletRV(alpha_0, mean)
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
#     return bayes_re(model_gen, bayes_model, rand_kwargs, model_kwargs)
#
#
#
#
# def bayes_re2(model_gen, bayes_model):
#
#     obj = model_gen(**bayes_model.rand_kwargs())
#     obj.bayes_model = bayes_model
#
#     def random_model(self):
#         for attr, val in bayes_model.rand_kwargs().items():
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
#                                   **{'supp': set_x_s['x'], 'set_y': set_y_s['y'], 'rng': rng})
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
#     bayes_model = DirichletRV(alpha_0, mean)
#     bayes_model.rand_kwargs = types.MethodType(rand_kwargs, bayes_model)
#
#     return bayes_re2(model_gen, bayes_model)








#%% Boneyard

# class BayesRE:
#     def __new__(cls, bayes_model, model_cls, model_kwargs):
#         self.bayes_model = bayes_model
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
#         cls.bayes_model = DirichletRV(2, [.5, .5])
#         p = cls.rng_prior()
#         return super().__new__(cls, supp, p, seed)
#
#     def __init__(self, supp, seed=None):
#         # self.bayes_model = DirichletRV(2, [.5, .5])
#         p = self.rng_prior()
#         super().__init__(supp, p, seed)
#
#     @classmethod
#     def rng_prior(cls):
#         return cls.bayes_model.rvs()
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
#     def __init__(self, bayes_model, model):
#         self.bayes_model = bayes_model
#         self.model = model
#
#     def random_model(self):
#         self.bayes_model.rvs()
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
