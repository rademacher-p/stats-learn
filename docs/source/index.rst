Welcome to Statistical Learning's documentation!
================================================
This package provides a framework to explore statistical learning with a Bayesian focus. The objective is to create and
apply prediction functions to the most common applications in machine learning: regression and classification.

This :mod:`stats_learn` package provides a framework to explore statistical learning with a Bayesian focus. It
implements a variety of
:mod:`random.elements`, as well as :mod:`random.models` of data for supervised learning. The :mod:`bayes` subpackage
implements similar elements/models with :func:`prior <stats_learn.bayes.elements.Base.prior>` attributes to statistically characterize parameter uncertainty and
:func:`fit <stats_learn.bayes.elements.Base.fit>` methods to adapt posteriors.

For supervised learning, the `predictors` subpackage provides objects that use these statistical models to define
inference and decision functions. Additionally, customization enables comparison with learning objects from popular
machine learning packages. The `predictors.torch` submodule uses `PyTorch <https://pytorch.org/>`_
(and `PyTorch Lightning <https://www.pytorchlightning.ai/>`_) to implement neural networks in the `stats_learn` API.

Also included (`results` submodule) are various functions that enable fair and reproducible evaluations, as well as
provide visualizations and Markdown-formatted output. Furthermore, they allow efficient assessments for learners
across a set of hyperparameter values.

Dat function is :class:`stats_learn.random.elements.Normal` with method :func:`stats_learn.random.elements.Normal.prob`

The :doc:`predictors <stats_learn.predictors>` are derived from statistical models
(i.e. probability distributions) of the joint :math:`\xrm` and :math:`\yrm` random elements. By defining the conditional
statistics :math:`\Prm(y|x)`, the predictor can operate on novel observations :math:`x` and make predictions about
unobserved values :math:`y`.

For non-learning predictors, :ref:`fixed models <random.models>` are used to define the model statistics
and thus the prediction function. For learning predictors, :ref:`Bayesian models <bayes.models>` provide
functionality for fitting posterior distributions and thus enabling adaptation of the prediction function to training
data :math:`\Drm = (\ldots, (\xrm_i, \yrm_i), \ldots)`

.. note::
   This project is under active development.

.. code-block::

   from stats_learn import random, bayes
   from stats_learn.predictors import ModelRegressor, BayesRegressor

   model = random.models.NormalLinear(weights=[1, 1])

   # Predictors
   opt_predictor = ModelRegressor(model, name='Optimal')

   norm_model = bayes.models.NormalLinear(prior_mean=[0, 0], prior_cov=1)
   norm_predictor = BayesRegressor(norm_model, name='Normal')

   # Results
   seed = 12345
   n_train = 10
   n_test = 20

   d = model.sample(n_train + n_test, rng=seed)
   d_train, d_test = d[:n_train], d[n_train:]

   loss_min = opt_predictor.evaluate(d_test)
   print(f"Minimum loss = {loss_min:.3f}")

   loss_prior = norm_predictor.evaluate(d_test)  # use the prior distribution
   print(f"Untrained learner loss = {loss_prior:.3f}")

   norm_predictor.fit(d_train)  # fit the posterior distribution
   loss_fit = norm_predictor.evaluate(d_test)
   print(f"Trained learner loss = {loss_fit:.3f}")

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   usage
   stats_learn


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
