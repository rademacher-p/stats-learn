"""
Bayesian SL models.
"""

import functools

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats._multivariate import _PSD

from thesis.random import elements as rand_elements
from thesis.random import models as rand_models

from thesis.util.base import RandomGeneratorMixin
from thesis.util import spaces

#%% Priors

# TODO: Add deterministic DEP to effect a DP realization and sample!!


class Base(RandomGeneratorMixin):
    def __init__(self, prior=None, rng=None):
        super().__init__(rng)

        self._space = {'x': None, 'y': None}

        self.prior = prior
        self.posterior = None
        self.posterior_model = None

    space = property(lambda self: self._space)

    shape = property(lambda self: {key: space.shape for key, space in self._space.items()})
    size = property(lambda self: {key: space.size for key, space in self._space.items()})
    ndim = property(lambda self: {key: space.ndim for key, space in self._space.items()})

    dtype = property(lambda self: {key: space.dtype for key, space in self._space.items()})

    def random_model(self, rng=None):
        raise NotImplementedError

    rvs = rand_elements.Base.rvs

    def _rvs(self, size, rng):
        model = self.random_model(rng)
        return model._rvs(size)

    def fit(self, d=None, warm_start=False):
        if d is None:
            d = np.array([], dtype=[(c, self.dtype[c], self.shape[c]) for c in 'xy'])

        self._fit(d, warm_start)

    def _fit(self, d, warm_start=False):
        raise NotImplementedError


class NormalRegressor(Base):
    def __init__(self, prior_mean=np.zeros(1), prior_cov=np.eye(1), basis_y_x=None, cov_y_x=1.,
                 model_x=rand_elements.Normal(),
                 rng=None):

        # Prior
        prior = rand_elements.Normal(prior_mean, prior_cov)
        super().__init__(prior, rng)
        if self.prior.ndim > 1:
            raise ValueError

        self._set_prior_persistent_attr()

        # Model
        self._set_model_x(model_x)
        self._set_cov_y_x(cov_y_x)
        self._set_basis_y_x(basis_y_x)

        # Learning
        self.posterior = rand_elements.Normal(self.prior_mean, self.prior_cov)
        self.posterior_model = rand_models.NormalRegressor(**self._prior_model_kwargs)

    # Methods
    def random_model(self, rng=None):
        rng = self._get_rng(rng)

        model_kwargs = {'basis_y_x': self.basis_y_x, 'cov_y_x': self.cov_y_x, 'model_x': self.model_x,
                        'rng': rng}
        rand_kwargs = {'weights': self.prior.rvs(rng=rng)}

        return rand_models.NormalRegressor(**model_kwargs, **rand_kwargs)

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
        self.posterior.mean = self.prior_mean
        self.posterior.cov = self.prior_cov

        for key, val in self._prior_model_kwargs.items():
            setattr(self.posterior_model, key, val)

    @property
    def _prior_model_kwargs(self):
        return {'weights': self.prior_mean, 'basis_y_x': self.basis_y_x, 'cov_y_x': self._prior_model_cov,
                'model_x': self.model_x}

    def _update_posterior(self, mean_only=False):
        if not mean_only:
            self.posterior.cov = np.linalg.inv(self._cov_prior_inv + self._cov_data_inv)
            self.posterior_model.cov_y_x_ = self._make_posterior_model_cov(self.posterior.cov)

        self.posterior.mean = self.posterior.cov @ (self._cov_prior_inv @ self.prior_mean + self._mean_data_temp)
        self.posterior_model.weights = self.posterior.mean

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
        self._space['x'] = val.space

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

            self._basis_y_x = tuple(power_func(i) for i in range(len(self.prior_mean)))
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
        self._space['y'] = spaces.Euclidean(_temp[:int(len(_temp) / 2)])

        self._prec_U_y_x = _PSD(self._cov_y_x.reshape(2 * (self.size['y'],)), allow_singular=False).U

    @cov_y_x.setter
    def cov_y_x(self, val):
        self._set_cov_y_x(val)
        self._reset_posterior()

    # Prior parameters
    @property
    def prior_mean(self):
        return self.prior.mean

    @prior_mean.setter
    def prior_mean(self, val):
        self.prior.mean = val
        self._update_posterior(mean_only=True)

    @property
    def prior_cov(self):
        return self.prior.cov

    @prior_cov.setter
    def prior_cov(self, val):
        self.prior.cov = val
        self._set_prior_persistent_attr()
        self._update_posterior()

    def _set_prior_persistent_attr(self):
        self._cov_prior_inv = np.linalg.inv(self.prior_cov)
        self._prior_model_cov = self._make_posterior_model_cov(self.prior_cov)


# def rsetattr(obj, attr, val):
#     pre, _, post = attr.rpartition('.')
#     # return setattr(rgetattr(obj, pre) if pre else obj, post, val)
#     return super().__setattr__(post, val)
#
#
# def rgetattr(obj, attr, *args):
#     def _getattr(obj_, attr_):
#         return getattr(obj_, attr_, *args)
#
#     return functools.reduce(_getattr, [obj] + attr.split('.'))


class Dirichlet(Base):
    def __init__(self, prior_mean, alpha_0, rng=None):
        super().__init__(prior=None, rng=rng)
        self._space = prior_mean.space

        _emp_dist = rand_models.DataEmpirical([], [], space=self.space)
        self.posterior_model = rand_models.Mixture([prior_mean, _emp_dist], [alpha_0, _emp_dist.n])

    def __repr__(self):
        return f"Dirichlet(alpha_0={self.alpha_0}, n={self.n}, prior_mean={self.prior_mean})"

    def __setattr__(self, name, value):
        if name.startswith('prior_mean.'):
            # setattr(self.prior_mean, name.removeprefix('prior_mean.'), value)
            self.posterior_model.set_dist_attr(0, **{name.removeprefix('prior_mean.'): value})
        else:
            super().__setattr__(name, value)

    @property
    def prior_mean(self):
        return self.posterior_model.dists[0]

    @prior_mean.setter
    def prior_mean(self, val):
        self.posterior_model.set_dist(0, val, self.alpha_0)

    @property
    def alpha_0(self):
        return self.posterior_model.weights[0]

    @alpha_0.setter
    def alpha_0(self, val):
        self.posterior_model.weights = [val, self.n]

    @property
    def emp_dist(self):
        return self.posterior_model.dists[1]

    @emp_dist.setter
    def emp_dist(self, val):
        self.posterior_model.set_dist(1, val, val.n)

    n = property(lambda self: self.emp_dist.n)

    def random_model(self, rng=None):
        raise NotImplementedError       # TODO: implement for finite in subclass?

    def _rvs(self, n, rng):
        # Samples directly from the marginal Dirichlet-Empirical data distribution

        _out = np.array([tuple(np.empty(self.shape[c], self.dtype[c]) for c in 'xy') for _ in range(n)],
                        dtype=[(c, self.dtype[c], self.shape[c]) for c in 'xy'])
        for i in range(n):
            if rng.random() <= self.alpha_0 / (self.alpha_0 + i):
                _out[i] = self.prior_mean.rvs(rng=rng)     # sample from mean distribution
            else:
                _out[i] = rng.choice(_out[:i])

        return _out

    def _fit(self, d, warm_start=False):
        if warm_start:
            emp_dist = self.emp_dist
        else:
            emp_dist = rand_models.DataEmpirical([], [], space=self.space)
        emp_dist.add_data(d)
        self.emp_dist = emp_dist


if __name__ == '__main__':
    theta = rand_models.NormalRegressor(weights=(1,), basis_y_x=(lambda x: x,), cov_y_x=.1,
                                        model_x=rand_elements.Finite(np.linspace(0, 1, 10, endpoint=False)),
                                        )
    alpha = rand_models.NormalRegressor(weights=(2,), basis_y_x=(lambda x: 1,), cov_y_x=.1,
                                        model_x=rand_elements.Finite(np.linspace(0, 1, 10, endpoint=False)),
                                        )

    # theta = rand_models.ClassConditional.from_finite([rand_elements.Normal(mean) for mean in (1, 3)], ['a', 'b'])
    # alpha = rand_models.ClassConditional.from_finite([rand_elements.Normal(mean) for mean in (0, 2)], ['a', 'b'])

    # theta = rand_models.ClassConditional.from_finite([rand_elements.Finite([1, 2], [p, 1 - p]) for p in (.1, .8)],
    #                                                  ['a', 'b'], p_y=None)
    # # alpha = rand_models.ClassConditional.from_finite([rand_elements.Finite([1, 2], [p, 1-p]) for p in (.3, .6)],
    # #                                                  ['a', 'b'], p_y=None)
    # alpha = rand_models.ClassConditional.from_finite([rand_elements.Finite([1, 2], [.5, .5]) for _ in range(2)],
    #                                                  ['a', 'b'], p_y=None)

    #
    x_plt = theta.space['x'].x_plt
    # ax_mode = theta.space['x'].make_axes()
    ax_mode = None

    # theta.plot_mode_y_x(ax=ax_mode)
    theta.plot_mean_y_x(ax=ax_mode)

    # print(f"Mode = {theta.mode}")
    # print(f"Mean = {theta.mean}")

    b = Dirichlet(prior_mean=alpha, alpha_0=1)
    # b.posterior_model.plot_mode_y_x(ax=ax_mode)
    b.posterior_model.plot_mean_y_x(ax=ax_mode)
    b.rvs(5)

    # print(f"Mode = {b.posterior_model.mode}")
    # print(f"Mean = {b.posterior_model.mean}")

    b.fit(theta.rvs(100))
    # b.posterior_model.plot_mode_y_x(ax=ax_mode)
    b.posterior_model.plot_mean_y_x(ax=ax_mode)
    b.rvs(10)

    # print(f"Mode = {b.posterior_model.mode}")
    # print(f"Mean = {b.posterior_model.mean}")

    b.alpha_0 = 2
