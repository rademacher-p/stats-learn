"""
Supervised learning functions.
"""

from abc import ABC, abstractmethod
from typing import Union
from functools import partial
from copy import deepcopy

import numpy as np

import sklearn as skl
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

import torch
from torch.utils.data import TensorDataset, DataLoader

from stats_learn.bayes import models as bayes_models
from stats_learn.loss_funcs import loss_se, loss_01
from stats_learn.random import elements as rand_elements, models as rand_models
from stats_learn.util import spaces
from stats_learn.util.base import vectorize_func

from stats_learn.util.results import (plot_fit_compare, predict_stats_compare, plot_predict_stats_compare,
                                      risk_eval_sim_compare, risk_eval_comp_compare, plot_risk_eval_sim_compare,
                                      plot_risk_eval_comp_compare)


class Base(ABC):
    def __init__(self, loss_func, proc_funcs=(), name=None):
        self.loss_func = loss_func

        self.proc_funcs = list(proc_funcs)
        self.name = name

        self.model = None

        self.can_warm_start = False

    # @property
    # def space(self):
    #     if self._space is None:
    #         self._space = self._model_obj.space
    #     return self._space
    # space = property(lambda self: self._space)

    space = property(lambda self: self._model_obj.space)

    shape = property(lambda self: {key: space.shape for key, space in self.space.items()})
    size = property(lambda self: {key: space.size for key, space in self.space.items()})
    ndim = property(lambda self: {key: space.ndim for key, space in self.space.items()})
    dtype = property(lambda self: {key: space.dtype for key, space in self.space.items()})

    @property
    @abstractmethod
    def _model_obj(self):
        raise NotImplementedError

    def set_params(self, **kwargs):  # TODO: improve? wrapper to ignore non-changing param set?
        for key, val in kwargs.items():
            setattr(self._model_obj, key, val)

    def tex_params(self, key, val=None):
        return self._model_obj.tex_params(key, val)

    # def get_params(self, *args):
    #     return {arg: getattr(self._model_obj, arg) for arg in args}

    def _proc_predictors(self, x):
        for func in self.proc_funcs:
            x = func(x)
        return x

    def _proc_data(self, d):
        x = self._proc_predictors(d['x'])
        dtype = [('x', d.dtype['x'].base, x.shape[1:]), ('y', d.dtype['y'].base, d.dtype['y'].shape)]
        return np.array(list(zip(x, d['y'])), dtype=dtype)

    def fit(self, d=None, warm_start=False):
        if d is None:
            d = np.array([], dtype=[(c, self.dtype[c], self.shape[c]) for c in 'xy'])

        d = self._proc_data(d)
        return self._fit(d, warm_start)

    @abstractmethod
    def _fit(self, d, warm_start):
        raise NotImplementedError

    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        d = model.rvs(n_train, rng=rng)  # generate train/test data
        self.fit(d, warm_start)  # train learner

    def predict(self, x):
        x = self._proc_predictors(x)
        return self._predict(x)

    def _predict(self, x):
        return vectorize_func(self._predict_single, shape=self.shape['x'])(x)

    def _predict_single(self, x):
        # raise NotImplementedError("Method must be overwritten.")  # TODO: numeric approx with loss and predictive!?
        pass

    def evaluate(self, d):
        loss = self.loss_func(self.predict(d['x']), d['y'], shape=self.shape['y'])
        return loss.mean()

    def evaluate_from_model(self, model, n_test=1, n_mc=1, rng=None):
        """Average empirical risk achieved from a given data model."""
        model.rng = rng
        loss = np.empty(n_mc)
        for i_mc in range(n_mc):
            d = model.rvs(n_test)  # generate train/test data
            loss[i_mc] = self.evaluate(d)  # make decision and assess

        return loss.mean()

    # Plotting utilities
    def plot_predict(self, x=None, ax=None, label=None):
        """Plot prediction function."""
        return self.space['x'].plot(self.predict, x, ax=ax, label=label)

    def plot_fit(self, d, ax=None):
        if ax is None:
            ax = self.space['x'].make_axes()
        return plot_fit_compare([self], d, ax)

    # def plot_fit(self, d, ax=None):
    #     if ax is None:
    #         ax = self.space['x'].make_axes()
    #     ax.scatter(d['x'], d['y'])
    #
    #     self.fit(d)
    #     self.plot_predict(ax=ax)

    # Prediction statistics
    def predict_stats(self, model=None, params=None, n_train=0, n_mc=1, x=None, stats=('mode',), verbose=False):
        if model is None:
            model = self._model_obj
        return predict_stats_compare([self], model, [params], n_train, n_mc, x, stats, verbose)[0]

    def plot_predict_stats(self, model=None, params=None, n_train=0, n_mc=1, x=None, do_std=False, verbose=False,
                           ax=None):
        if model is None:
            model = self._model_obj
        return plot_predict_stats_compare([self], model, [params], n_train, n_mc, x, do_std, verbose, ax)

    # Risk evaluation
    def risk_eval_sim(self, model=None, params=None, n_train=0, n_test=1, n_mc=1, verbose=False):
        if model is None:
            model = self._model_obj
        return risk_eval_sim_compare([self], model, [params], n_train, n_test, n_mc, verbose)[0]

    def plot_risk_eval_sim(self, model=None, params=None, n_train=0, n_test=1, n_mc=1, verbose=False, ax=None):
        if model is None:
            model = self._model_obj
        return plot_risk_eval_sim_compare([self], model, [params], n_train, n_test, n_mc, verbose, ax)

    def risk_eval_comp(self, model=None, params=None, n_train=0, n_test=1, verbose=False):
        if model is None:
            model = self._model_obj
        return risk_eval_comp_compare([self], model, [params], n_train, n_test, verbose)[0]

    def plot_risk_eval_comp(self, model=None, params=None, n_train=0, n_test=1, verbose=False, ax=None):
        if model is None:
            model = self._model_obj
        return plot_risk_eval_comp_compare([self], model, [params], n_train, n_test, verbose, ax)


class ClassifierMixin:
    model: rand_models.Base

    def _predict(self, x):
        return self.model.mode_y_x(x)


class RegressorMixin:
    model: Union[rand_models.Base, rand_models.MixinRVy]

    def _predict(self, x):
        return self.model.mean_y_x(x)


#%% Fixed model
class Model(Base):
    def __init__(self, model, loss_func, proc_funcs=(), name=None):
        super().__init__(loss_func, proc_funcs, name)
        self.model = model

    def __repr__(self):
        return self.__class__.__name__ + f"(model={self.model})"

    @property
    def _model_obj(self):
        return self.model

    def _fit(self, d, warm_start):
        pass

    def fit_from_model(self, model, n_train=0, rng=None, warm_start=False):
        pass  # skip unnecessary data generation


class ModelClassifier(ClassifierMixin, Model):
    def __init__(self, model, proc_funcs=(), name=None):
        super().__init__(model, loss_01, proc_funcs, name)


class ModelRegressor(RegressorMixin, Model):
    def __init__(self, model, proc_funcs=(), name=None):
        super().__init__(model, loss_se, proc_funcs, name)

    def evaluate_comp(self, model=None, n_train=0, n_test=1):
        if model is None:
            model = self._model_obj

        n_train = np.array(n_train)

        if isinstance(model, (rand_models.Base, rand_models.MixinRVy)):
            if isinstance(model.space['x'], spaces.FiniteGeneric):
                x = model.space['x'].values_flat

                p_x = model.model_x.pf(x)

                cov_y_x = model.cov_y_x(x)
                bias_sq = (self.predict(x) - model.mean_y_x(x)) ** 2

                risk = np.dot(cov_y_x + bias_sq, p_x)
                return np.full(n_train.shape, risk)
            else:
                raise NotImplementedError

        elif isinstance(model, bayes_models.Base):
            raise NotImplementedError


#%% Bayes model

class Bayes(Base):
    def __init__(self, bayes_model, loss_func, proc_funcs=(), name=None):
        super().__init__(loss_func, proc_funcs, name=name)

        self.bayes_model = bayes_model

        self.can_warm_start = self.bayes_model.can_warm_start

        self.prior = self.bayes_model.prior
        self.posterior = self.bayes_model.posterior

        self.model = self.bayes_model.posterior_model  # updates in-place with set_params() and fit()

        self.fit()

    def __repr__(self):
        return self.__class__.__name__ + f"(bayes_model={self.bayes_model})"

    @property
    def _model_obj(self):
        return self.bayes_model

    def _fit(self, d, warm_start):
        self.bayes_model.fit(d, warm_start)

    def plot_param_dist(self, x=None, ax_prior=None):  # TODO: improve or remove?
        if x is None:
            raise ValueError  # TODO

        self.prior.plot_pf(x, ax=ax_prior)
        # ax_posterior= plt_prior.axes
        # ax_posterior = plt.gca()
        ax_posterior = None
        self.posterior.plot_pf(x, ax=ax_posterior)


class BayesClassifier(ClassifierMixin, Bayes):
    def __init__(self, bayes_model, proc_funcs=(), name=None):
        super().__init__(bayes_model, loss_01, proc_funcs, name)


class BayesRegressor(RegressorMixin, Bayes):
    def __init__(self, bayes_model, proc_funcs=(), name=None):
        super().__init__(bayes_model, loss_se, proc_funcs, name)

    def evaluate_comp(self, model=None, n_train=0, n_test=1):
        if model is None:
            model = self._model_obj

        n_train = np.array(n_train)

        if isinstance(model, (rand_models.Base, rand_models.MixinRVy)):
            if (isinstance(model.space['x'], spaces.FiniteGeneric)
                    and isinstance(self.bayes_model, bayes_models.Dirichlet)):

                x = model.space['x'].values_flat

                p_x = model.model_x.pf(x)
                alpha_x = self.bayes_model.alpha_0 * self.bayes_model.prior_mean.model_x.pf(x)

                cov_y_x = model.cov_y_x(x)
                bias_sq = (self.bayes_model.prior_mean.mean_y_x(x) - model.mean_y_x(x)) ** 2

                w_cov = np.zeros((n_train.size, p_x.size))
                w_bias = np.zeros((n_train.size, p_x.size))
                for i_n, n_i in enumerate(n_train.flatten()):
                    # rv = rand_elements.EmpiricalScalar(n_i, .5)
                    rv = rand_elements.Binomial(.5, n_i)
                    supp = rv.space.values
                    for i_x, (p_i, a_i) in enumerate(zip(p_x, alpha_x)):
                        rv.p = p_i
                        p_rv = rv.pf(supp)

                        # den = (a_i + n_i * supp) ** 2
                        den = (a_i + supp) ** 2

                        # w_cov[i_n, i_x] = (p_rv / den * n_i * supp).sum()
                        w_cov[i_n, i_x] = (p_rv / den * supp).sum()
                        w_bias[i_n, i_x] = (p_rv / den * a_i ** 2).sum()

                risk = np.dot(cov_y_x * (1 + w_cov) + bias_sq * w_bias, p_x)

                return risk.reshape(n_train.shape)
            else:
                raise NotImplementedError

        elif isinstance(model, bayes_models.Base):

            if (isinstance(model.space['x'], spaces.FiniteGeneric)
                    and isinstance(self.bayes_model, bayes_models.Dirichlet)):

                if (isinstance(model, bayes_models.Dirichlet) and model.alpha_0 == self.bayes_model.alpha_0
                        and model.prior_mean == self.bayes_model.prior_mean and n_test == 1):
                    # Minimum Bayesian squared-error

                    x = model.space['x'].values_flat

                    alpha_0 = self.bayes_model.alpha_0
                    alpha_m = self.bayes_model.prior_mean.model_x.pf(x)
                    weights = (alpha_m + 1 / (alpha_0 + n_train[..., np.newaxis])) / (alpha_m + 1 / alpha_0)

                    # return (alpha_m * weights * self.bayes_model.prior_mean.cov_y_x(x)).sum(axis=-1)
                    return np.dot(weights * self.bayes_model.prior_mean.cov_y_x(x), alpha_m)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError


class SKLWrapper(Base):

    # FIXME: inheritance feels broken

    def __init__(self, estimator, space, proc_funcs=(), name=None):
        if skl.base.is_regressor(estimator):
            loss_func = loss_se
        else:
            raise ValueError("Estimator must be regressor-type.")

        super().__init__(loss_func, proc_funcs, name)
        self.estimator = estimator
        self._space = space

    space = property(lambda self: self._space)

    @property
    def _model_obj(self):
        raise NotImplementedError

    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self.estimator, key, val)

    def _fit(self, d, warm_start):
        if hasattr(self.estimator, 'warm_start'):  # TODO: check unneeded if not warm_start
            self.estimator.set_params(warm_start=warm_start)
        elif isinstance(self.estimator, Pipeline):
            self.estimator.set_params(regressor__warm_start=warm_start)  # assumes pipeline step called "regressor"
        else:
            raise NotImplementedError

        if len(d) > 0:
            x, y = d['x'].reshape(-1, 1), d['y']
            self.estimator.fit(x, y)
        elif not warm_start:
            self.estimator = skl.base.clone(self.estimator)  # manually reset learner if `fit` is not called

    def _predict(self, x):
        try:
            x = x.reshape(-1, 1)
            return self.estimator.predict(x)
        except NotFittedError:
            return np.full(x.shape[0], np.nan)


def reset_weights(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class LitWrapper(Base):  # TODO: move to submodule to avoid excess imports
    def __init__(self, model, trainer, space, proc_funcs=(), name=None):
        loss_func = loss_se  # TODO: Generalize!

        super().__init__(loss_func, proc_funcs, name)
        self.model = model
        self.trainer = trainer
        self._trainer_init = deepcopy(self.trainer)
        self._space = space

    space = property(lambda self: self._space)

    @property
    def _model_obj(self):
        raise NotImplementedError

    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self.model, key, val)

    def reset(self):  # TODO: add reset method to predictor base class?
        self.model.apply(reset_weights)

        # self.trainer.current_epoch = 0
        self.trainer = deepcopy(self._trainer_init)

    def _reshape_batches(self, *arrays):
        shape_x = self.shape['x']
        if shape_x == ():  # cast to non-scalar shape
            shape_x = (1,)
        return tuple(map(lambda x: x.reshape(-1, *shape_x), arrays))

    def _fit(self, d, warm_start):
        if not warm_start:
            self.reset()

        x, y = self._reshape_batches(d['x'], d['y'])
        x, y = map(partial(torch.tensor, dtype=torch.float32), (x, y))
        ds = TensorDataset(x, y)

        batch_size = len(x)  # TODO: no mini-batching! Allow user specification.
        dl = DataLoader(ds, batch_size, shuffle=True)

        self.trainer.fit(self.model, dl)

    def _predict(self, x):
        x, = self._reshape_batches(x)
        x = torch.tensor(x, requires_grad=False, dtype=torch.float32)
        y_hat = self.model(x).detach().numpy()
        y_hat = y_hat.reshape(-1, *self.shape['y'])
        return y_hat
