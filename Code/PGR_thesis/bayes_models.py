"""
Bayesian Prior objects.
"""

import math

import numpy as np
from scipy.stats._multivariate import _PSD

from random_elements import Normal, Dirichlet
import RE_obj_callable
from models import DataConditional, NormalRegressor as NormalRegressorModel
from util.generic import RandomGeneratorMixin, empirical_pmf
from util.func_obj import FiniteDomainFunc

#%% Priors

# TODO: Add deterministic DEP to effect a DP realization and sample!!


class Base(RandomGeneratorMixin):
    # param_names = ()

    def __init__(self, prior=None, rng=None):
        super().__init__(rng)

        self._shape = {'x': None, 'y': None}

        self.prior = prior
        self.posterior = None
        self.posterior_model = None

    shape = property(lambda self: self._shape)
    size = property(lambda self: {key: math.prod(val) for key, val in self._shape.items()})
    ndim = property(lambda self: {key: len(val) for key, val in self._shape.items()})

    def random_model(self, rng=None):
        raise NotImplementedError

    def fit(self, d=None, warm_start=False):
        if d is None:
            d = np.array([], dtype=[('x', '<f8', self.shape['x']),
                                    ('y', '<f8', self.shape['y'])])

        self._fit(d, warm_start)

    def _fit(self, d, warm_start=False):
        raise NotImplementedError

    # @property
    # def posterior(self):
    #     return self._posterior
    #
    # @property
    # def posterior_model(self):
    #     return self._posterior_model


# class Base:
#     def __init__(self, model_cls, model_kwargs=None, prior=None, rng=None):
#         self._shape = {'x': None, 'y': None}
#
#         if model_kwargs is None:
#             self.model_kwargs = {}
#         else:
#             self.model_kwargs = model_kwargs
#
#         self.model_cls = model_cls
#         self.prior = prior
#
#         self.posterior = None
#         self.posterior_model = None
#
#         self.rng = rng      # setter binds RNG to model and prior
#
#     shape = property(lambda self: self._shape)
#     size = property(lambda self: {key: math.prod(val) for key, val in self._shape.items()})
#     ndim = property(lambda self: {key: len(val) for key, val in self._shape.items()})
#
#     @property
#     def rng(self):
#         return self._rng
#
#     @rng.setter
#     def rng(self, rng):
#         self._rng = check_rng(rng)
#         self.model_kwargs.update(rng=self._rng)
#         self.prior.rng = self._rng
#
#     def random_model(self, rng=None):
#         model_kwargs = self.model_kwargs.copy()
#         if rng is None:
#             rand_kwargs = self.random_kwargs()
#         else:
#             rng = check_rng(rng)
#             model_kwargs.update(rng=rng)
#             rand_kwargs = self.random_kwargs(rng)
#
#         return self.model_cls(**model_kwargs, **rand_kwargs)   # defaults to deterministic bayes_model?
#
#     def random_kwargs(self, rng=None):
#         raise NotImplementedError
#
#     def fit(self, d=None, warm_start=False):
#         if d is None:
#             d = np.array([], dtype=[('x', '<f8', self.shape['x']),
#                                     ('y', '<f8', self.shape['y'])])
#
#         self._fit(d, warm_start)
#
#     def _fit(self, d, warm_start=False):
#         raise NotImplementedError


# class BetaModelBayes(Base):
#     def __init__(self, prior=None, rng_model=None):     # deterministic
#         model_gen = DataConditional.beta_model
#         model_kwargs = {'a': .9, 'b': .9, 'c': 5, 'rng': rng_model}
#         super().__init__(model_gen, model_kwargs, prior)
#
#         self._data_shape_x = ()
#         self._data_shape_y = ()


class NormalRegressor(Base):
    # param_names = ('model_x', 'basis_y_x', 'cov_y_x', 'mean_prior', 'cov_prior')

    def __init__(self, model_x=Normal(), basis_y_x=None, cov_y_x=1.,
                 mean_prior=np.zeros(1), cov_prior=np.eye(1), rng=None):

        # Prior
        prior = Normal(mean_prior, cov_prior)
        super().__init__(prior, rng)
        if self.prior.ndim > 1:
            raise ValueError

        self._set_prior_persistent_attr()

        # Model
        self._set_model_x(model_x)
        self._set_cov_y_x(cov_y_x)
        self._set_basis_y_x(basis_y_x)

        # Learning
        self.posterior = Normal(self.mean_prior, self.cov_prior)
        self.posterior_model = NormalRegressorModel(**self._prior_model_kwargs)

    # Methods
    def random_model(self, rng=None):
        rng = self._get_rng(rng)

        model_kwargs = {'model_x': self.model_x, 'basis_y_x': self.basis_y_x,
                        'cov_y_x_single': self.cov_y_x, 'rng': rng}
        rand_kwargs = {'weights': self.prior.rvs(rng=rng)}

        return NormalRegressorModel(**model_kwargs, **rand_kwargs)

    def _fit(self, d, warm_start=False):
        if not warm_start:  # reset learning attributes
            self._cov_data_inv = np.zeros(2 * self.prior.shape)
            self._mean_data_temp = np.zeros(self.prior.shape)

        n = len(d)
        if n > 0:  # update data-dependent attributes
            psi = np.array([np.array([func(x_i) for func in self.basis_y_x])
                            for x_i in d['x']]).reshape((n, self.prior.size, self.size['y']))
            psi_white = np.dot(psi, self._prec_U_y_x)
            self._cov_data_inv += sum(psi_i @ psi_i.T for psi_i in psi_white)

            y_white = np.dot(d['y'].reshape(n, self.size['y']), self._prec_U_y_x)
            self._mean_data_temp += sum(psi_i @ y_i for psi_i, y_i in zip(psi_white, y_white))

        self._update_posterior()

    def _reset_posterior(self):
        self.posterior.mean = self.mean_prior
        self.posterior.cov = self.cov_prior

        for key, val in self._prior_model_kwargs.items():
            setattr(self.posterior_model, key, val)

    def _update_posterior(self, mean_only=False):
        if not mean_only:
            self.posterior.cov = np.linalg.inv(self._cov_prior_inv + self._cov_data_inv)
            self.posterior_model.cov_y_x_single = self._make_posterior_model_cov(self.posterior.cov)

        self.posterior.mean = self.posterior.cov @ (self._cov_prior_inv @ self.mean_prior + self._mean_data_temp)
        self.posterior_model.weights = self.posterior.mean

    @property
    def _prior_model_kwargs(self):
        return {'model_x': self.model_x, 'basis_y_x': self.basis_y_x,
                'cov_y_x_single': self._prior_model_cov, 'weights': self.mean_prior}

    def _make_posterior_model_cov(self, cov_weight):
        def cov_y_x(x):
            psi_x = np.array([func(x) for func in self.basis_y_x]).reshape(self.prior.size, self.size['y'])
            return self.cov_y_x + (psi_x.T @ cov_weight @ psi_x).reshape(2 * self.shape['y'])

        return cov_y_x

    # Model parameters
    @property
    def model_x(self):
        return self._model_x

    def _set_model_x(self, val):
        self._model_x = val
        self._shape['x'] = val.shape

    @model_x.setter
    def model_x(self, val):
        self._set_model_x(val)
        self._reset_posterior()

    @property
    def basis_y_x(self):
        return self._basis_y_x

    def _set_basis_y_x(self, val):
        if val is None:
            def power_func(i):
                return lambda x: np.full(self.shape['y'], (x ** i).sum())

            self._basis_y_x = tuple(power_func(i) for i in range(len(self.mean_prior)))
        else:
            self._basis_y_x = val

    @basis_y_x.setter
    def basis_y_x(self, val):
        self._set_basis_y_x(val)
        self._reset_posterior()

    @property
    def cov_y_x(self):
        return self._cov_y_x

    def _set_cov_y_x(self, val):
        self._cov_y_x = np.array(val)

        _temp = self._cov_y_x.shape
        self._shape['y'] = _temp[:int(len(_temp) / 2)]

        self._prec_U_y_x = _PSD(self._cov_y_x.reshape(2 * (self.size['y'],)), allow_singular=False).U

    @cov_y_x.setter
    def cov_y_x(self, val):
        self._set_cov_y_x(val)
        self._reset_posterior()

    # Prior parameters
    @property
    def mean_prior(self):
        return self.prior.mean

    @mean_prior.setter
    def mean_prior(self, val):
        self.prior.mean = val
        self._update_posterior(mean_only=True)

    @property
    def cov_prior(self):
        return self.prior.cov

    @cov_prior.setter
    def cov_prior(self, val):
        self.prior.cov = val
        self._set_prior_persistent_attr()
        self._update_posterior()

    def _set_prior_persistent_attr(self):
        self._cov_prior_inv = np.linalg.inv(self.prior.cov)
        self._prior_model_cov = self._make_posterior_model_cov(self.prior.cov)





# class NormalRegressor(Base):
#     def __init__(self, model_x=Normal(), basis_y_x=None, cov_y_x=1.,
#                  mean_prior=np.zeros(1), cov_prior=np.eye(1), rng=None):
#
#         prior = Normal(mean_prior, cov_prior, rng=None)
#
#         self.cov_y_x = np.array(cov_y_x)
#         _temp = self.cov_y_x.shape
#         _shape_y = _temp[:int(len(_temp) / 2)]
#
#         self.model_x = model_x
#
#         if basis_y_x is None:
#             def power_func(i):
#                 return lambda x: np.full(_shape_y, (x**i).sum())
#             self.basis_y_x = tuple(power_func(i) for i in range(len(prior.mean)))
#         else:
#             self.basis_y_x = basis_y_x
#
        # model_cls = NormalRegressorModel
        # model_kwargs = {'model_x': self.model_x, 'basis_y_x': self.basis_y_x,
        #                 'cov_y_x_single': self.cov_y_x, 'rng': None}
        # super().__init__(model_cls, model_kwargs, prior, rng)
#
#         self._shape = {'x': model_x.shape, 'y': _shape_y}
#
#         # Persistent attributes
#         _psd = _PSD(self.cov_y_x.reshape(2 * (self.size['y'],)), allow_singular=False)
#         self._prec_U_y_x = _psd.U
#         self._cov_prior_inv = np.linalg.inv(self.cov_prior)
#
#         self._reset()
#
#     @property
#     def mean_prior(self):
#         return self.prior.mean
#
#     @mean_prior.setter
#     def mean_prior(self, val):
#         self.prior.mean = val
#
#     @property
#     def cov_prior(self):
#         return self.prior.cov
#
#     @cov_prior.setter
#     def cov_prior(self, val):
#         self.prior.cov = val
#         self._cov_prior_inv = np.linalg.inv(self.prior.cov)
#
#     # @property
#     # def cov_y_x(self):
#     #     pass
#
#     # def random_model(self, rng=None):
#     #     model_kwargs = self.model_kwargs.copy()     # TODO: need copy?
#     #     if rng is None:
#     #         rand_kwargs = self.random_kwargs()
#     #     else:
#     #         rng = check_rng(rng)
#     #         model_kwargs.update(rng=rng)
#     #         rand_kwargs = self.random_kwargs(rng)
#     #
#     #     return self.model_cls(**model_kwargs, **rand_kwargs)   # defaults to deterministic bayes_model?
#
#     def random_kwargs(self, rng=None):
#         return {'weights': self.prior.rvs(rng=rng)}
#
#     def _reset(self):
#         """Reset learning attributes"""
#         self._cov_post_inv = self._cov_prior_inv.copy()
#         self._mean_post_temp = (self._cov_post_inv @ self.mean_prior).copy()
#
#         self.posterior = self.prior
#         self.posterior_model = self._post_model(self.posterior.mean, self.posterior.cov)    # TODO: posterior setter!?
#
#     def _fit(self, d, warm_start=False):
#         if not warm_start:
#             self._reset()
#
#         n = len(d)
#         if n == 0:
#             return
#
#         # Update posterior
#         psi = np.array([np.array([func(x_i) for func in self.basis_y_x])
#                         for x_i in d['x']]).reshape((n, self.prior.size, self.size['y']))
#         psi_white = np.dot(psi, self._prec_U_y_x)
#         self._cov_post_inv += sum(psi_i @ psi_i.T for psi_i in psi_white)
#
#         y_white = np.dot(d['y'].reshape(n, self.size['y']), self._prec_U_y_x)
#         self._mean_post_temp += sum(psi_i @ y_i for psi_i, y_i in zip(psi_white, y_white))
#
#         cov_post = np.linalg.inv(self._cov_post_inv)
#         mean_post = cov_post @ self._mean_post_temp
#
#         self.posterior = Normal(mean_post, cov_post)
#
#         # Update model
#
#         # def cov_y_x(x):
#         #     psi_x = np.array([func(x) for func in self.basis_y_x]).reshape(self.prior.size, self.size['y'])
#         #     return self.cov_y_x + (psi_x.T @ self.posterior.cov @ psi_x).reshape(2 * self.shape['y'])
#         #
#         # kwargs = self.model_kwargs.copy()
#         # kwargs.update(weights=self.posterior.mean, cov_y_x=cov_y_x, rng=None)
#         #
#         # self.posterior_model = NormalRegressorModel(**kwargs)
#
#         self.posterior_model = self._post_model(self.posterior.mean, self.posterior.cov)
#
#     def _post_model(self, mean, cov):
#         def cov_y_x(x):
#             psi_x = np.array([func(x) for func in self.basis_y_x]).reshape(self.prior.size, self.size['y'])
#             return self.cov_y_x + (psi_x.T @ cov @ psi_x).reshape(2 * self.shape['y'])
#
#         kwargs = self.model_kwargs.copy()
#         kwargs.update(weights=mean, cov_y_x=cov_y_x, rng=None)
#
#         return NormalRegressorModel(**kwargs)
#
#     # def posterior(self, d):
#     #     n = len(d)
#     #     if n == 0:
#     #         return self.prior
#     #     else:
#     #         psi = np.array([np.array([func(x_i) for func in self.basis_y_x])
#     #                         for x_i in d['x']]).reshape((n, self.prior.size, self.size['y']))
#     #
#     #         psi_white = np.dot(psi, self._prec_U_y_x)
#     #         cov_post_inv = self._cov_prior_inv + sum(psi_i @ psi_i.T for psi_i in psi_white)
#     #         cov_post = np.linalg.inv(cov_post_inv)
#     #
#     #         y_white = np.dot(d['y'].reshape(-1, self.size['y']), self._prec_U_y_x)
#     #         _temp = sum(psi_i @ y_i for psi_i, y_i in zip(psi_white, y_white))
#     #         mean_post = cov_post @ (self._cov_prior_inv @ self.mean_prior + _temp)
#     #
#     #         return RE_obj.Normal(mean_post, cov_post)
#     #
#     # def _posterior_2_model(self, posterior):
#     #     def cov_y_x(x):
#     #         psi_x = np.array([func(x) for func in self.basis_y_x]).reshape(self.prior.size, self.size['y'])
#     #         return self.cov_y_x + (psi_x.T @ posterior.cov @ psi_x).reshape(2 * self.shape['y'])
#     #
#     #     kwargs = self.model_kwargs.copy()
#     #     kwargs.update(weights=posterior.mean, cov_y_x=cov_y_x, rng=None)
#     #
#     #     return NormalRegressor(**kwargs)
#     #
#     # def posterior_model(self, d):
#     #     posterior = self.posterior(d)
#     #     return self._posterior_2_model(posterior)
#     #
#     # def fit(self, d):
#     #     posterior = self.posterior(d)
#     #     # predictive_dist = self._posterior_2_predictive(posterior)
#     #     # posterior_model = self._predictive_2_model(predictive_dist)
#     #     posterior_model = self._posterior_2_model(posterior)
#     #     return posterior, posterior_model
#
#     # def _posterior_2_predictive(self, posterior):
#     #     def model_y_x(x):
#     #         mean_y_x = sum(weight * func(x) for weight, func in zip(posterior.mean, self.basis_y_x))
#     #         # psi_x = np.array([func(x) for func in self.basis_y_x]).T
#     #         # cov_y_x = self.cov_y_x + inner_prod(psi_x.T, psi_x.T, posterior.cov)    # TODO
#     #         psi_x = np.array([func(x) for func in self.basis_y_x]).reshape(self.prior.size, self.size['y'])
#     #         cov_y_x = self.cov_y_x + (psi_x.T @ posterior.cov @ psi_x).reshape(2 * self.shape['y'])
#     #
#     #         return RE_obj.Normal(mean_y_x, cov_y_x)
#     #
#     #     return model_y_x
#
#     # def predictive_dist(self, d):
#     #     posterior = self.posterior(d)
#     #     return self._posterior_2_predictive(posterior)
#
#     # def _predictive_2_model(self, predictive_dist):
#     #     return DataConditional(model_x=self.model_kwargs['model_x'], model_y_x=predictive_dist)


# bayes_model = NormalRegressor(basis_y_x=None, cov_y_x=1., mean_prior=np.zeros(2), cov_prior=np.eye(2), rng=100)
# for _ in range(2):
#     model = bayes_model.random_model(rng=None)
#     print(model.weights)
#     rvs = model.rvs(2, rng=None)
#     print(rvs)


#%% TODO FIXME: rework, fix shape attributes...

class DirichletFiniteYcXModelBayesNew(Base):
    def __init__(self, alpha_0, mean_x, mean_y_x, rng_model=None, rng_prior=None):
        model_gen = DataConditional.finite_model
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
        p_x = self.prior['p_x'].rvs(rng=rng)

        val = []        # FIXME: better way?
        for x_flat in self._mean_x._supp_flat:
            x = x_flat.reshape(self._data_shape_x)
            val.append(self.prior['p_y_x'](x).rvs(rng=rng))
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

        def p_y_x(x):
            x = np.asarray(x)
            d_match = d[np.all(x.flatten() == d['x'].reshape(n, -1), axis=-1)].squeeze()
            n_match = d_match.size
            if n_match == 0:
                return self._mean_y_x(x)

            c_prior_y = 1 / (1 + n_match / (self.alpha_0 * self._mean_x(x)))
            emp_dist = FiniteDomainFunc(self._supp_y, empirical_pmf(d_match['y'], self._supp_y, self._data_shape_y))
            return c_prior_y * self._mean_y_x(x) + (1 - c_prior_y) * emp_dist

        # return p_y_x

        def model_y_x(x):
            return RE_obj_callable.FiniteRE(p_y_x(x))
        return model_y_x



#%% Without func objects
class FiniteYcXModelBayes(Base):
    def __init__(self, supp_x, supp_y, prior, rng_model=None):
        model_gen = DataConditional.finite_model_orig
        # model_kwargs = {'rng': rng_model}
        model_kwargs = {'supp_x': supp_x['x'], 'supp_y': supp_y['y'], 'rng': rng_model}
        super().__init__(model_gen, model_kwargs, prior)

        self.supp_x = supp_x
        self.supp_y = supp_y        # TODO: Assumed to be my SL structured array!

        self._supp_shape_x = supp_x.shape
        self._supp_shape_y = supp_y.shape
        self._data_shape_x = supp_x.dtype['x'].shape
        self._data_shape_y = supp_y.dtype['y'].shape

    def random_model(self):
        raise NotImplementedError("Method must be overwritten.")


class DirichletFiniteYcXModelBayes(FiniteYcXModelBayes):

    # TODO: initialization from marginal/conditional? full mean init as classmethod?

    def __init__(self, supp_x, supp_y, alpha_0, mean, rng_model=None, rng_prior=None):
        prior = Dirichlet(alpha_0, mean, rng_prior)
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

        # self._model_gen = functools.partial(DataConditional.finite_model, supp_x=supp_x['x'], supp_y=supp_y['y'], rng=None)

    def random_model(self, rng=None):
        p = self.prior.rvs(rng=rng)

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
# class FiniteYcXModelBayesOrig(Base):
#     def __init__(self, supp_x, supp_y, prior, rng_model=None):     # TODO: rng check, None default?
#         self.supp_x = supp_x
#         self.supp_y = supp_y        # TODO: Assumed to be my SL structured array!
#
#         self._supp_shape_x = supp_x.shape
#         self._supp_shape_y = supp_y.shape
#         self._data_shape_x = supp_x.dtype['x'].shape
#         self._data_shape_y = supp_y.dtype['y'].shape
#
#         model_gen = DataConditional.finite_model_orig
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
#         prior = Dirichlet(alpha_0, mean, rng_prior)
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
#         self._model_gen = functools.partial(DataConditional.finite_model_orig,
#                                             supp_x=supp_x['x'], supp_y=supp_y['y'], rng=None)
#
#     def random_model(self, rng=None):
#         p = self.prior.rvs(rng=rng)
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
#     def posterior_model(self, d):
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
#         bayes_model = Dirichlet(alpha_0, mean)
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
#         model_gen = DataConditional.finite_model
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
#     model_gen = DataConditional.finite_model
#     model_kwargs = {'supp': set_x_s['x'], 'set_y': set_y_s['y'], 'rng': rng}
#
#     bayes_model = Dirichlet(alpha_0, mean)
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
#     model_gen = functools.partial(DataConditional.finite_model,
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
#     bayes_model = Dirichlet(alpha_0, mean)
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
#         cls.bayes_model = Dirichlet(2, [.5, .5])
#         p = cls.rng_prior()
#         return super().__new__(cls, supp, p, seed)
#
#     def __init__(self, supp, seed=None):
#         # self.bayes_model = Dirichlet(2, [.5, .5])
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
# dist = Dirichlet(4, [.5, .5])
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
# dist = Dirichlet(4, [.5, .5])
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
#         theta = FiniteRE(self.support, theta_pmf, self.rng)
#         # theta_m = FiniteRE(self.support_x, theta_pmf_m, self.rng)
#
#         return theta
#
#     @classmethod
#     def dirichlet_prior(cls, alpha_0, mean, support, seed=None):
#         return cls(Dirichlet(alpha_0, mean), support, seed)
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
#         return DataConditional(theta_m, theta_c)
