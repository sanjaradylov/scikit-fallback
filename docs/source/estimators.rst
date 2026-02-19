Estimators
==========

The ``skfb.estimators`` module implements *fallback meta-estimators* that extend classification
models with a reject option. They offer configurations allowing a model to refuse to make predictions
(and return fallback labels), accept all inputs but mask rejections in background, or ignore rejections.
Can be used in high-stakes environments, where instead of accepting all inputs, the model needs to
delegate potential anomalies or uncertainties to other specialists.

Threshold-Based Rejectors
-------------------------

The following fallback estimators accept and/or learn hard rules for rejection, e.g., based on certainty
thresholds.

.. autoclass:: skfb.estimators.ThresholdFallbackClassifier
    :inherited-members: fit, predict, predict_proba, predict_log_proba, decision_function

.. autoclass:: skfb.estimators.ThresholdFallbackClassifierCV
    :inherited-members: fit, predict, predict_proba, predict_log_proba, decision_function

.. autoclass:: skfb.estimators.MultiThresholdFallbackClassifier
    :inherited-members: fit, predict, predict_proba, predict_log_proba, decision_function

.. autoclass:: skfb.estimators.RateFallbackClassifierCV
    :inherited-members: fit, predict, predict_proba, predict_log_proba, decision_function

.. autoclass:: skfb.estimators.RuleClassifier
    :inherited-members: fit, predict

.. autoclass:: skfb.estimators.FallbackRuleClassifier
    :inherited-members: fit, predict

.. autofunction:: skfb.estimators.predict_or_fallback

.. autofunction:: skfb.estimators.multi_threshold_predict_or_fallback

Anomaly-Based Fallback Classifiers
----------------------------------

These fallback estimators reject based on outlier/novelty classification.

.. autoclass:: skfb.estimators.AnomalyFallbackClassifier
    :members:
    :inherited-members: predict
