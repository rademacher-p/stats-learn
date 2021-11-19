.. Statistical Learning documentation master file, created by
   sphinx-quickstart on Fri Nov 12 11:56:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Statistical Learning's documentation!
================================================
This package provides a framework to explore statistical learning with a Bayesian focus. The objective is to create and
apply prediction functions to the most common applications in machine learning: regression and classification.

A function is :py:class:`stats_learn.random.elements.Normal`

The :doc:`predictors <stats_learn.predictors>` are derived from statistical models
(i.e. probability distributions) of the joint :math:`\xrm` and :math:`\yrm` random elements. By defining the conditional
statistics :math:`\Prm(y|x)`, the predictor can operate on novel observations :math:`x` and make predictions about
unobserved values :math:`y`.

For non-learning predictors, :doc:`fixed models <stats_learn.random>` are used to define the model statistics
and thus the prediction function. For learning predictors, :doc:`Bayesian models <stats_learn.bayes>` provide
functionality for fitting posterior distributions and thus enabling adaptation of the prediction function to training
data :math:`\Drm = (\ldots, (\xrm_i, \yrm_i), \ldots)`

.. note::
   This project is under active development at https://github.com/rademacher-p/stats-learn

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   usage
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
