"""The :mod:`skfb.estimators` module implements fallback meta-estimators."""

__all__ = (
    "multi_threshold_predict_or_fallback",
    "predict_or_fallback",
    "AnomalyFallbackClassifier",
    "FallbackRuleClassifier",
    "MultiThresholdFallbackClassifier",
    "RateFallbackClassifierCV",
    "RuleClassifier",
    "ThresholdFallbackClassifier",
    "ThresholdFallbackClassifierCV",
)

from ._anomaly import AnomalyFallbackClassifier

from ._multi_threshold import (
    multi_threshold_predict_or_fallback,
    MultiThresholdFallbackClassifier,
)

from ._rule import FallbackRuleClassifier, RuleClassifier

from ._threshold import (
    predict_or_fallback,
    RateFallbackClassifierCV,
    ThresholdFallbackClassifier,
    ThresholdFallbackClassifierCV,
)
