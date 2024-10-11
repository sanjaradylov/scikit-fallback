Estimators
==========

The ``skfb.estimators`` module implements fallback meta-estimators.

Threshold-Based Rejectors
-------------------------

The following fallback estimators accept and/or learn the certainty threshold(s) for
rule-based rejection.

.. autoclass:: skfb.estimators.ThresholdFallbackClassifier
    :inherited-members:

.. autoclass:: skfb.estimators.ThresholdFallbackClassifierCV
    :inherited-members:

.. autoclass:: skfb.estimators.MultiThresholdFallbackClassifier
    :inherited-members:

.. autoclass:: skfb.estimators.RateFallbackClassifierCV
    :inherited-members:


Anomaly-Based Fallback Classifiers
----------------------------------

These fallback estimators reject based on outlier/novelty classification.

.. autoclass:: skfb.estimators.AnomalyFallbackClassifier
    :inherited-members:
