import types
from RE_obj import FiniteRE, DirichletRV


#%% Priors

def bayes_re(model_cls, prior, rand_kwargs, model_kwargs={}):
    prior.rand_kwargs = types.MethodType(rand_kwargs, prior)

    obj = model_cls(**model_kwargs, **prior.rand_kwargs())
    obj.prior = prior

    def random_model(self):
        for attr, val in prior.rand_kwargs().items():
            setattr(self, attr, val)
    obj.random_model = types.MethodType(random_model, obj)

    return obj

model_cls = FiniteRE
model_kwargs = {'supp': ['a', 'b']}

prior = DirichletRV(4, [.4, .6])
def rand_kwargs(self): return {'p': self.rvs()}

c = bayes_re(model_cls, prior, rand_kwargs, model_kwargs)
c.random_model()




def bayes_re1(model_cls, model_kwargs, prior, func):
    obj = model_cls(**model_kwargs, **func(prior))
    obj.prior = prior

    def rng_model(self):
        for attr, val in func(self.prior).items():
            setattr(self, attr, val)
    obj.random_model = types.MethodType(rng_model, obj)

    return obj


model_cls = FiniteRE
model_kwargs = {'supp': ['a', 'b']}

prior = DirichletRV(4, [.4, .6])
def func(obj): return {'p': obj.rvs()}
# def func(self): return {'p': self.rvs()}

a = bayes_re1(model_cls, model_kwargs, prior, func)
a.random_model()




def bayes_re2(model_cls, model_kwargs, prior):
    obj = model_cls(**model_kwargs, **prior.func())
    obj.prior = prior

    def rng_model(self):
        for attr, val in prior.func().items():
            setattr(self, attr, val)
    obj.random_model = types.MethodType(rng_model, obj)

    return obj

model_cls = FiniteRE
model_kwargs = {'supp': ['a', 'b']}

prior = DirichletRV(4, [.4, .6])
def func(self): return {'p': self.rvs()}
prior.func = types.MethodType(func, prior)

b = bayes_re2(model_cls, model_kwargs, prior)
b.random_model()








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
