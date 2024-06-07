"""The :mod:`skfb.estimators` module implements fallback meta-estimators."""

__all__ = (
    "predict_or_fallback",
    "RateFallbackClassifier",
    "ThresholdFallbackClassifier",
    "ThresholdFallbackClassifierCV",
)

from ._threshold import (
    predict_or_fallback,
    RateFallbackClassifier,
    ThresholdFallbackClassifier,
    ThresholdFallbackClassifierCV,
)
