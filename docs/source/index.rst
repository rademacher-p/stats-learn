Statistical Learning package documentation
==========================================

.. note::
   This project is under active development.

The :mod:`stats_learn` package provides a framework to explore statistical learning with a Bayesian focus. See :doc:`usage` to get started and :doc:`stats_learn` to explore the API.

A variety of :ref:`random elements <random.elements>` :math:`\mathrm{x} \in \mathcal{X}` are implemented in the :mod:`random.elements <stats_learn.random.elements>` submodule. Similar to other `distributions <https://pytorch.org/docs/stable/distributions.html>`_ packages, methods are available for random sampling (:func:`sample <stats_learn.random.elements.Base.sample>`) and probability evaluation (:func:`prob <stats_learn.random.elements.Base.prob>`). Additionally, statistics such as the :func:`mode <stats_learn.random.elements.Base.mode>` and :func:`mean <stats_learn.random.elements.MixinRV.mean>` can be accessed.

For supervised learning analysis, :ref:`random models <random.models>` are implemented in the :mod:`random.models <stats_learn.random.models>` submodule. These models define fixed joint distributions :math:`\mathrm{P}_{\mathrm{y},\mathrm{x}}` over the observed :ref:`random elements <random.elements>` :math:`\mathrm{x} \in \mathcal{X}` and the unobserved elements :math:`\mathrm{y} \in \mathcal{Y}`. Conditional random elements can be generated with the :func:`model_y_x <stats_learn.random.models.Base.model_y_x>` method and used for prediction; furthermore, conditional statistics such as :func:`mode_y_x <stats_learn.random.models.Base.mode_y_x>` are directly available.

The :mod:`bayes <stats_learn.bayes>` subpackage implements similar elements/models with parametric representations. :ref:`Bayesian models <bayes.models>` define data distributions :math:`\mathrm{P}_{\mathrm{y},\mathrm{x} | \uptheta}` and use a :func:`prior <stats_learn.bayes.elements.Base.prior>` :math:`\mathrm{P}_{\uptheta}` to characterize the model uncertainty. Using observed training data pairs :math:`\mathrm{D} = (\ldots, (\mathrm{x}_i, \mathrm{y}_i), \ldots)`, the :func:`fit <stats_learn.bayes.elements.Base.fit>` method formulates the posterior :math:`\mathrm{P}_{\uptheta | \mathrm{D}}` and the resultant Bayesian data model :math:`\mathrm{P}_{\mathrm{y},\mathrm{x} | \mathrm{D}}`.

To deploy statistical models in supervised learning applications, the :mod:`predictors <stats_learn.predictors>` subpackage provides objects that use these models to define inference and decision functions (:func:`fit <stats_learn.predictors.base.Base.fit>` and :func:`predict <stats_learn.predictors.base.Base.predict>`, respectively). By forming the predictive distribution :math:`\mathrm{P}_{\mathrm{y} | \mathrm{x}}` (or :math:`\mathrm{P}_{\mathrm{y} | \mathrm{x}, \mathrm{D}}` for Bayesian models), these :doc:`predictors <stats_learn.predictors>` can operate on novel observations :math:`\mathrm{x}` and generate decisions :math:`h \in \mathcal{H}` for arbitrary loss functions :math:`L: \mathcal{H} \times \mathcal{Y} \mapsto \mathbb{R}_{\geq 0}`. To enable comparison with learning objects from popular machine learning packages, the API allows straightforward customization of predictor subclasses. The :mod:`predictors.torch <stats_learn.predictors.torch>` submodule uses `PyTorch <https://pytorch.org/>`_ (and `PyTorch Lightning <https://www.pytorchlightning.ai/>`_) to implement neural networks in the :mod:`stats_learn` API.

The package also includes the :mod:`results <stats_learn.results>` submodule, which provides various functions that enable fair and reproducible evaluations, as well as provide visualizations and Markdown-formatted output. The :func:`data_assess <stats_learn.results.data_assess>` function allows evaluation against a fixed dataset; the :func:`model_assess <stats_learn.results.model_assess>` function effects Monte Carlo approximation of the expected loss by using a specified model to randomly generate datasets for evaluation. These functions also allow each learner to be assessed over a set of hyperparameter values, performing computationally efficient retraining when possible.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   stats_learn


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
