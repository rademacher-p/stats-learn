Welcome to Statistical Learning's documentation!
================================================

.. note::
   This project is under active development.

The :mod:`stats_learn` package provides a framework to explore statistical learning with a Bayesian focus. It
implements a variety of
:mod:`random.elements <stats_learn.random.elements>`, as well as :mod:`random.models <stats_learn.random.models>` of
data for supervised learning. The :mod:`bayes <stats_learn.bayes>` subpackage implements similar elements/models with
:func:`prior <stats_learn.bayes.elements.Base.prior>` attributes to statistically characterize parameter uncertainty
and :func:`fit <stats_learn.bayes.elements.Base.fit>` methods to adapt posteriors.

For supervised learning, the :mod:`predictors <stats_learn.predictors>` subpackage provides objects that use these
statistical models to define inference and decision functions. Additionally, customization enables comparison with
learning objects from popular machine learning packages. The :mod:`predictors.torch <stats_learn.predictors.torch>`
submodule uses `PyTorch <https://pytorch.org/>`_ (and `PyTorch Lightning <https://www.pytorchlightning.ai/>`_) to
implement neural networks in the :mod:`stats_learn` API.

Also included (in the :mod:`results <stats_learn.results>` submodule) are various functions that enable fair and
reproducible evaluations, as well as provide visualizations and Markdown-formatted output. Furthermore, they allow
efficient assessments for learners across a set of hyperparameter values.

The :doc:`predictors <stats_learn.predictors>` are derived from statistical models
(i.e. probability distributions) of the joint :math:`\xrm` and :math:`\yrm` random elements. By defining the conditional
statistics :math:`\Prm(y|x)`, the predictor can operate on novel observations :math:`x` and make predictions about
unobserved values :math:`y`.

For non-learning predictors, :ref:`fixed models <random.models>` are used to define the model statistics
and thus the prediction function. For learning predictors, :ref:`Bayesian models <bayes.models>` provide
functionality for fitting posterior distributions and thus enabling adaptation of the prediction function to training
data :math:`\Drm = (\ldots, (\xrm_i, \yrm_i), \ldots)` is :py:`d`

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
