"""Fixed and learning predictors for supervised learning applications."""

from abc import ABC, abstractmethod
from functools import partial
from operator import itemgetter

import numpy as np

from stats_learn import bayes, random, results, spaces
from stats_learn.loss_funcs import loss_01, loss_se
from stats_learn.util import vectorize_func


# Base and Mixin classes
class Base(ABC):
    r"""
    Base class for supervised learning predictors.

    Parameters
    ----------
    loss_func : callable
    space : dict, optional
        The domain for :math:`\mathrm{x}` and :math:`\mathrm{y}`. Defaults to the
        model's space.
    proc_funcs : Collection of callable of dict of Collection of callable
        Sequentially-invoked preprocessing functions for :math:`x` and :math:`y` values.
    name : str, optional

    """

    def __init__(self, loss_func, space=None, proc_funcs=(), name=None):
        self.loss_func = loss_func

        self._space = space

        if isinstance(proc_funcs, dict):
            self.proc_funcs = proc_funcs
        else:
            self.proc_funcs = {"pre": list(proc_funcs), "post": []}
        self.name = str(name)

        self.model = None  # `random.models.Base` object used to generate predictions

        self.can_warm_start = False  # enables incremental training

    @property
    def space(self):
        r"""The domain for :math:`\mathrm{x}` and :math:`\mathrm{y}`.

        Defaults to the model's space.
        """
        if self._space is None:
            self._space = self._model_obj.space
        return self._space

    shape = property(
        lambda self: {key: space.shape for key, space in self.space.items()}
    )
    size = property(lambda self: {key: space.size for key, space in self.space.items()})
    ndim = property(lambda self: {key: space.ndim for key, space in self.space.items()})
    dtype = property(
        lambda self: {key: space.dtype for key, space in self.space.items()}
    )

    @property
    @abstractmethod
    def _model_obj(self):
        raise NotImplementedError

    # TODO: improve? wrapper to ignore non-changing param set?
    def set_params(self, **kwargs):
        """Set parameters of the learning model object."""
        for key, value in kwargs.items():
            setattr(self._model_obj, key, value)

    def tex_params(self, key, value=None):
        return self._model_obj.tex_params(key, value)

    def _proc_x(self, x):
        for func in self.proc_funcs["pre"]:
            x = func(x)
        return x

    def _proc_y(self, y):
        for func in self.proc_funcs["post"]:
            y = func(y)
        return y

    def _proc_data(self, d):
        x, y = self._proc_x(d["x"]), self._proc_y(d["y"])
        dtype = [
            ("x", d.dtype["x"].base, x.shape[1:]),
            ("y", d.dtype["y"].base, y.shape[1:]),
        ]
        return np.array(list(zip(x, y)), dtype=dtype)

    def fit(self, d=None, warm_start=False):
        """
        Refine the learning model using observations.

        Parameters
        ----------
        d : np.ndarray, optional
            The training data.
        warm_start : bool, optional
            If `False`, `reset` is invoked to restore unfit state.

        """
        if not warm_start:
            self.reset()
        elif not self.can_warm_start:
            raise ValueError("Predictor does not support warm start fitting.")

        if d is not None and len(d) > 0:
            d = self._proc_data(d)
            self._fit(d)

    @abstractmethod
    def _fit(self, d):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Restore unfit prior state."""
        raise NotImplementedError

    def fit_from_model(self, model, n_train=0, warm_start=False, rng=None):
        """
        Refine the learning model using data randomly drawn from a model.

        Parameters
        ----------
        model : stats_learn.random.models.Base
            Model for training data generation.
        n_train : int, optional
            Number of training samples.
        warm_start : bool, optional
            If `False`, `reset` is invoked to restore unfit state.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        """
        d = model.sample(n_train, rng=rng)  # generate train/test data
        self.fit(d, warm_start)  # train learner

    def predict(self, x):
        r"""
        Generate predictions for given :math:`x` values.

        Parameters
        ----------
        x : array_like
            Observed random element values.

        Returns
        -------
        np.ndarray
            Prediction values.

        """
        x = self._proc_x(x)
        y = self._predict(x)
        y = self._proc_y(y)
        return y

    # @abstractmethod  # TODO
    # def _predict(self, x):
    #     raise NotImplementedError

    def _predict(self, x):
        vec_func = vectorize_func(self._predict_single, self.shape["x"])
        return vec_func(x)

    def _predict_single(self, x):
        model_y = self.model.model_y_x(x)

        # TODO: cache predictions?
        def _risk(h):  # TODO: memoize here?
            _fn = partial(self.loss_func, h)
            return model_y.expectation(_fn)

        space_h = self.space["y"]

        # TODO: generalize and make argument for convex closure
        if isinstance(space_h, spaces.FiniteGeneric):
            vals = space_h.values_flat
            lims = vals.min(axis=0), vals.max(axis=0)
            space_h = spaces.Box(lims)

        return space_h.argmin(_risk)

    def evaluate(self, loss_func, d):
        """
        Evaluate predictor using test data.

        Parameters
        ----------
        loss_func: callable
        d : np.ndarray
            The test data.

        Returns
        -------
        float
            Empirical risk (i.e. average test loss).

        """
        losses = loss_func(self.predict(d["x"]), d["y"], shape=self.shape["y"])
        return losses.mean()

    def evaluate_from_model(self, loss_func, model, n_test=1, n_mc=1, rng=None):
        """
        Evaluate predictor using test data randomly drawn from a given data model.

        Parameters
        ----------
        loss_func: callable
        model : stats_learn.random.models.Base
            Model for training data generation.
        n_test : int, optional
            Number of test samples.
        n_mc : int, optional
            Number of Monte Carlo simulation iterations.
        rng : int or np.random.RandomState or np.random.Generator, optional
            Random number generator seed or object.

        Returns
        -------
        float
            Empirical risk (i.e. average test loss).

        """
        model.rng = rng
        losses = np.empty(n_mc)
        for i_mc in range(n_mc):
            d = model.sample(n_test)
            losses[i_mc] = self.evaluate(loss_func, d)

        return losses.mean()

    # Plotting utilities
    def plot_predict(self, x=None, ax=None, label=None):
        """
        Plot prediction function.

        Parameters
        ----------
        x : array_like, optional
            Values to plot against. Defaults to `self.x_plt`.
        ax : matplotlib.axes.Axes, optional
            Axes.
        label : str, optional
            Label for matplotlib.artist.Artist

        Returns
        -------
        matplotlib.artist.Artist or tuple of matplotlib.artist.Artist

        """
        return self.space["x"].plot(self.predict, x, ax=ax, label=label)

    # Assessment
    def data_assess(
        self,
        loss_func,
        d_train=None,
        d_test=None,
        params=None,
        x=None,
        verbose=False,
        plot_fit=False,
        log_path=None,
        img_path=None,
        ax=None,
    ):
        """
        Assess predictor using a single dataset.

        Parameters
        ----------
        loss_func : callable
        d_train : array_like, optional
            Training data.
        d_test : array_like, optional
            Testing data.
        params : dict, optional
            Predictor parameters to evaluate. Outer product of each parameter array is
            assessed.
        x : array_like, optional
            Values of observed element to use for assessment of prediction statistics.
        verbose : bool, optional
            Enables iteration print-out.
        plot_fit : bool, optional
            Enables plotting of fit predictors.
        log_path : os.PathLike or str, optional
            File for saving printed loss table and image path in Markdown format.
        img_path : os.PathLike or str, optional
            Directory for saving generated images.
        ax : matplotlib.axes.Axes, optional
            Axes onto which stats/losses are plotted.

        Returns
        -------
        list of ndarray
            Empirical risk values for each parameterization.

        """
        return results.data_assess(
            [self],
            loss_func,
            d_train,
            d_test,
            [params],
            x,
            verbose,
            plot_fit,
            log_path,
            img_path,
            ax,
        )

    def model_assess(
        self,
        loss_func,
        model=None,
        params=None,
        n_train=0,
        n_test=0,
        n_mc=1,
        x=None,
        stats=None,
        verbose=False,
        plot_stats=False,
        plot_loss=False,
        print_loss=False,
        log_path=None,
        img_path=None,
        ax=None,
        rng=None,
    ):
        """
        Assess predictor via prediction statistics and empirical risk.

        Uses Monte Carlo simulation.

        Parameters
        ----------
        loss_func : callable
        model : stats_learn.random.models.Base or stats_learn.bayes.models.Base
            Data-generating model.
        params : Collection of dict, optional
            Predictor parameters to evaluate. Outer product of each parameter array is
            assessed.
        n_train : int or Collection of int, optional
            Training data volume.
        n_test : int, optional
            Test data volume.
        n_mc : int, optional
            Number of Monte Carlo simulation iterations.
        x : array_like, optional
            Values of observed element to use for assessment of prediction statistics.
        stats : Collection of str, optional
            Names of the statistics to generate, e.g. 'mean', 'std', 'cov', 'mode', etc.
        verbose : bool, optional
            Enables iteration print-out.
        plot_stats : bool, optional
            Enables plotting of prediction statistics.
        plot_loss : bool, optional
            Enables plotting of average loss.
        print_loss : bool, optional
            Enables print-out of average loss table.
        log_path : os.PathLike or str, optional
            File for saving printed loss table and image path in Markdown format.
        img_path : os.PathLike or str, optional
            Directory for saving generated images.
        ax : matplotlib.axes.Axes, optional
            Axes onto which stats/losses are plotted.
        rng : int or np.random.RandomState or np.random.Generator, optional
                Random number generator seed or object.

        Returns
        -------
        ndarray
            Prediction statistics for each parameterization.
        ndarray
            Empirical risk values for each parameterization.

        """
        if model is None:
            model = self._model_obj

        out = results.model_assess(
            [self],
            loss_func,
            model,
            [params],
            n_train,
            n_test,
            n_mc,
            x,
            stats,
            verbose,
            plot_stats,
            plot_loss,
            print_loss,
            log_path,
            img_path,
            ax,
            rng,
        )
        return map(itemgetter(0), out)

    # Analytical evaluation
    def risk_eval_analytic(
        self, model=None, params=None, n_train=0, n_test=1, verbose=False
    ):
        if model is None:
            model = self._model_obj
        return results.risk_eval_analytic(
            [self], model, [params], n_train, n_test, verbose
        )[0]


class ClassifierMixin:
    """Uses model conditional mode to minimize 0-1 loss."""

    model: random.models.Base

    def _predict(self, x):
        return self.model.mode_y_x(x)


class RegressorMixin:
    """Uses model conditional mean to minimize squared-error loss."""

    model: random.models.Base | random.models.MixinRVy

    def _predict(self, x):
        return self.model.mean_y_x(x)


# Static predictors using fixed data models
class Model(Base):
    r"""
    Predictor based on fixed data model.

    Parameters
    ----------
    model : stats_learn.random.models.Base
        Fixed model used to generate predictions.
    loss_func : callable
    space : dict, optional
        The domain for :math:`\mathrm{x}` and :math:`\mathrm{y}`. Defaults to the
        model's space.
    proc_funcs : Collection of callable of dict of Collection of callable
        Sequentially-invoked preprocessing functions for :math:`x` and :math:`y` values.
    name : str, optional

    """

    def __init__(self, model, loss_func, space=None, proc_funcs=(), name=None):
        super().__init__(loss_func, space, proc_funcs, name)
        self.model = model

    def __repr__(self):
        return self.__class__.__name__ + f"(model={self.model})"

    @property
    def _model_obj(self):
        return self.model

    def _fit(self, d):
        pass

    def reset(self):
        pass

    def fit_from_model(self, model, n_train=0, warm_start=False, rng=None):
        pass  # skip unnecessary data generation


class ModelClassifier(ClassifierMixin, Model):
    r"""
    Classifier based on fixed data model.

    Parameters
    ----------
    model : stats_learn.random.models.Base
        Fixed model used to generate predictions.
    space : dict, optional
        The domain for :math:`\mathrm{x}` and :math:`\mathrm{y}`. Defaults to the
        model's space.
    proc_funcs : Collection of callable of dict of Collection of callable
        Sequentially-invoked preprocessing functions for :math:`x` and :math:`y` values.
    name : str, optional

    """

    def __init__(self, model, space=None, proc_funcs=(), name=None):
        super().__init__(model, loss_01, space, proc_funcs, name)


class ModelRegressor(RegressorMixin, Model):
    r"""
    Regressor based on fixed data model.

    Parameters
    ----------
    model : stats_learn.random.models.Base
        Fixed model used to generate predictions.
    space : dict, optional
        The domain for :math:`\mathrm{x}` and :math:`\mathrm{y}`. Defaults to the
        model's space.
    proc_funcs : Collection of callable of dict of Collection of callable
        Sequentially-invoked preprocessing functions for :math:`x` and :math:`y` values.
    name : str, optional

    """

    def __init__(self, model, space=None, proc_funcs=(), name=None):
        super().__init__(model, loss_se, space, proc_funcs, name)

    def evaluate_analytic(self, model=None, n_train=0, n_test=1):
        """
        Analytically calculate SE risk.

        Parameters
        ----------
        model : stats_learn.random.models.Base
            Model for training data generation.
        n_train : int, optional
            Number of training samples.
        n_test : int, optional
            Number of testing samples.

        Returns
        -------
        float
            Analytical risk.

        """
        if model is None:
            model = self._model_obj

        n_train = np.array(n_train)

        if isinstance(model, random.models.BaseRVy):
            if isinstance(model.space["x"], spaces.FiniteGeneric):
                x = model.space["x"].values_flat

                p_x = model.model_x.prob(x)

                cov_y_x = model.cov_y_x(x)
                bias_sq = (self.predict(x) - model.mean_y_x(x)) ** 2

                risk = np.dot(cov_y_x + bias_sq, p_x)
                return np.full(n_train.shape, risk)
            else:
                raise NotImplementedError

        elif isinstance(model, bayes.models.Base):
            raise NotImplementedError


# Learning predictors using Bayesian data models
class Bayes(Base):
    r"""
    Predictor based on Bayesian data model.

    Parameters
    ----------
    bayes_model : stats_learn.bayes.models.Base
        Bayes model used for fitting and to generate predictions.
    loss_func : callable
    space : dict, optional
        The domain for :math:`\mathrm{x}` and :math:`\mathrm{y}`. Defaults to the
        model's space.
    proc_funcs : Collection of callable of dict of Collection of callable
        Sequentially-invoked preprocessing functions for :math:`x` and :math:`y` values.
    name : str, optional

    """

    def __init__(self, bayes_model, loss_func, space=None, proc_funcs=(), name=None):
        super().__init__(loss_func, space, proc_funcs, name=name)

        self.bayes_model = bayes_model

        self.can_warm_start = self.bayes_model.can_warm_start

        self.prior = self.bayes_model.prior
        self.posterior = self.bayes_model.posterior

        # model updates in-place with set_params() and fit()
        self.model = self.bayes_model.posterior_model

        self.fit()

    def __repr__(self):
        return self.__class__.__name__ + f"(bayes_model={self.bayes_model})"

    @property
    def _model_obj(self):
        return self.bayes_model

    def _fit(self, d):
        self.bayes_model.fit(d, warm_start=True)

    def reset(self):
        """Invoke reset of the Bayesian model."""
        self.bayes_model.reset()


class BayesClassifier(ClassifierMixin, Bayes):
    r"""
    Classifier based on Bayesian data model.

    Parameters
    ----------
    bayes_model : stats_learn.bayes.models.Base
        Bayes model used for fitting and to generate predictions.
    space : dict, optional
        The domain for :math:`\mathrm{x}` and :math:`\mathrm{y}`. Defaults to the
        model's space.
    proc_funcs : Collection of callable of dict of Collection of callable
        Sequentially-invoked preprocessing functions for :math:`x` and :math:`y` values.
    name : str, optional

    """

    def __init__(self, bayes_model, space=None, proc_funcs=(), name=None):
        super().__init__(bayes_model, loss_01, space, proc_funcs, name)


class BayesRegressor(RegressorMixin, Bayes):
    r"""
    Regressor based on Bayesian data model.

    Parameters
    ----------
    bayes_model : stats_learn.bayes.models.Base
        Bayes model used for fitting and to generate predictions.
    space : dict, optional
        The domain for :math:`\mathrm{x}` and :math:`\mathrm{y}`. Defaults to the
        model's space.
    proc_funcs : Collection of callable of dict of Collection of callable
        Sequentially-invoked preprocessing functions for :math:`x` and :math:`y` values.
    name : str, optional

    """

    def __init__(self, bayes_model, space=None, proc_funcs=(), name=None):
        super().__init__(bayes_model, loss_se, space, proc_funcs, name)

    def evaluate_analytic(self, model=None, n_train=0, n_test=1):
        """
        Analytically calculate SE risk.

        Parameters
        ----------
        model : stats_learn.random.models.Base
            Model for training data generation.
        n_train : int, optional
            Number of training samples.
        n_test : int, optional
            Number of testing samples.

        Returns
        -------
        float
            Analytical risk.

        """
        if model is None:
            model = self._model_obj

        n_train = np.array(n_train)

        prior_mean = self.bayes_model.prior_mean
        if isinstance(model, random.models.Base | random.models.MixinRVy):
            if isinstance(model.space["x"], spaces.FiniteGeneric) and isinstance(
                self.bayes_model, bayes.models.Dirichlet
            ):
                x = model.space["x"].values_flat

                p_x = model.model_x.prob(x)
                alpha_x = self.bayes_model.alpha_0 * prior_mean.model_x.prob(x)

                cov_y_x = model.cov_y_x(x)
                bias_sq = (prior_mean.mean_y_x(x) - model.mean_y_x(x)) ** 2

                w_cov = np.zeros((n_train.size, p_x.size))
                w_bias = np.zeros((n_train.size, p_x.size))
                for i_n, n_i in enumerate(n_train.flatten()):
                    rv = random.elements.Binomial(0.5, n_i)
                    values = rv.space.values
                    for i_x, (p_i, a_i) in enumerate(zip(p_x, alpha_x)):
                        rv.p = p_i
                        p_rv = rv.prob(values)

                        den = (a_i + values) ** 2

                        w_cov[i_n, i_x] = (p_rv / den * values).sum()
                        w_bias[i_n, i_x] = (p_rv / den * a_i**2).sum()

                risk = np.dot(cov_y_x * (1 + w_cov) + bias_sq * w_bias, p_x)

                return risk.reshape(n_train.shape)
            else:
                raise NotImplementedError

        elif isinstance(model, bayes.models.Base):
            if isinstance(model.space["x"], spaces.FiniteGeneric) and isinstance(
                self.bayes_model, bayes.models.Dirichlet
            ):
                if (
                    isinstance(model, bayes.models.Dirichlet)
                    and model.alpha_0 == self.bayes_model.alpha_0
                    and model.prior_mean == self.bayes_model.prior_mean
                    and n_test == 1
                ):
                    # Minimum Bayesian squared-error

                    x = model.space["x"].values_flat

                    alpha_0 = self.bayes_model.alpha_0
                    alpha_m = self.bayes_model.prior_mean.model_x.prob(x)
                    weights = (alpha_m + 1 / (alpha_0 + n_train[..., np.newaxis])) / (
                        alpha_m + 1 / alpha_0
                    )

                    return np.dot(weights * prior_mean.cov_y_x(x), alpha_m)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
