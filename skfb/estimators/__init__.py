"""The :mod:`skfb.estimators` module implements fallback meta-estimators."""

__all__ = (
    "predict_or_fallback",
    "RateFallbackClassifierCV",
    "ThresholdFallbackClassifier",
    "ThresholdFallbackClassifierCV",
)

from ._threshold import (
    predict_or_fallback,
    RateFallbackClassifierCV,
    ThresholdFallbackClassifier,
    ThresholdFallbackClassifierCV,
)
