"""The :mod:`skfb.estimators` module implements fallback meta-estimators."""

__all__ = (
    "multi_threshold_predict_or_fallback",
    "predict_or_fallback",
    "AnomalyFallbackClassifier",
    "CoverageFallbackClassifierCV",
    "FallbackRuleClassifier",
    "MultiThresholdFallbackClassifier",
    "RateFallbackClassifierCV",
    "RuleClassifier",
    "ThresholdFallbackClassifier",
    "ThresholdFallbackClassifierCV",
    "UtilityFallbackClassifierCV",
)

from ._anomaly import AnomalyFallbackClassifier

from ._multi_threshold import (
    multi_threshold_predict_or_fallback,
    MultiThresholdFallbackClassifier,
)

from ._rule import FallbackRuleClassifier, RuleClassifier

from ._threshold import (
    predict_or_fallback,
    CoverageFallbackClassifierCV,
    RateFallbackClassifierCV,
    ThresholdFallbackClassifier,
    ThresholdFallbackClassifierCV,
    UtilityFallbackClassifierCV,
)
