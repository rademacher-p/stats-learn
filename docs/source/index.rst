.. Statistical Learning documentation master file, created by
   sphinx-quickstart on Fri Nov 12 11:56:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Statistical Learning's documentation!
================================================
This package provides a framework to explore statistical learning with a Bayesian focus. The objective is to create and
apply prediction functions to the most common applications in machine learning: regression and classification.

This package provides a framework to explore statistical learning with a Bayesian focus. It implements a variety of
`random.elements`, as well as `random.models` of data for supervised learning. The `bayes` subpackage
implements similar elements/models with `prior` attributes to statistically characterize parameter uncertainty and
`fit` methods to adapt posteriors.

For supervised learning, the `predictors` subpackage provides objects that use these statistical models to define
inference and decision functions. Additionally, customization enables comparison with learning objects from popular
machine learning packages. The `predictors.torch` submodule uses [PyTorch](https://pytorch.org/)
(and [PyTorch Lightning](https://www.pytorchlightning.ai/)) to implement neural networks in the `stats_learn` API.

Also included (`results` submodule) are various functions that enable fair and reproducible evaluations, as well as
provide visualizations and Markdown-formatted output. Furthermore, they allow efficient assessments for learners
across a set of hyperparameter values.

Dat function is :py:class:`stats_learn.random.elements.Normal`

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

   from stats_learn import results

   predictors = [opt_predictor, norm_predictor]
   params = [None, {'prior_cov': [.01, .1, 1]}]

   # Sample regressor realizations
   results.data_assess(predictors, d_train, d_test, params, verbose=True, plot_fit=True)

   # Prediction mean/variance
   results.model_assess(predictors, model, params, n_train, n_test, n_mc=10, stats=('mean', 'std'), verbose=True,
                        plot_stats=True, print_loss=True, rng=seed)

   # Squared-Error vs. training data volume
   n_train = range(0, 100, 5)
   results.model_assess(predictors, model, params, n_train, n_test, n_mc=10, verbose=True, plot_loss=True, rng=seed)

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
