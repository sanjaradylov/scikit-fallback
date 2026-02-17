Ensemble Methods
================

The ``skfb.ensemble`` module implements cascade and routing ensemble methods for
efficient and selective multi-model inference.

Threshold Cascade Classifiers
------------------------------

These ensemble estimators learn confidence thresholds for efficient cascading decisions.

.. autoclass:: skfb.ensemble.ThresholdCascadeClassifier
    :inherited-members: fit, predict, predict_proba

.. autoclass:: skfb.ensemble.ThresholdCascadeClassifierCV
    :inherited-members: fit, predict, predict_proba

Routing Classifiers
-------------------

.. autoclass:: skfb.ensemble.RoutingClassifier
    :inherited-members: fit, predict, predict_proba

Exceptions and Warnings
-----------------------

.. autoclass:: skfb.ensemble.CascadeNotFittedWarning

.. autoclass:: skfb.ensemble.CascadeParetoConfigWarning

.. autoclass:: skfb.ensemble.CascadeParetoConfigException
