Estimators
==========

The ``skfb.estimators`` module implements fallback meta-estimators.

Threshold-Based Rejectors
-------------------------

The following fallback estimators accept and/or learn the certainty threshold(s) for
rule-based rejection.

.. autoclass:: skfb.estimators.ThresholdFallbackClassifier
    :inherited-members: fit, predict, predict_proba, predict_log_proba, decision_function

.. autoclass:: skfb.estimators.ThresholdFallbackClassifierCV
    :inherited-members: fit, predict, predict_proba, predict_log_proba, decision_function

.. autoclass:: skfb.estimators.MultiThresholdFallbackClassifier
    :inherited-members: fit, predict, predict_proba, predict_log_proba, decision_function

.. autoclass:: skfb.estimators.RateFallbackClassifierCV
    :inherited-members: fit, predict, predict_proba, predict_log_proba, decision_function


Anomaly-Based Fallback Classifiers
----------------------------------

These fallback estimators reject based on outlier/novelty classification.

.. autoclass:: skfb.estimators.AnomalyFallbackClassifier
    :members:
    :inherited-members: predict
