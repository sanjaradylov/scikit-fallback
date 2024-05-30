"""The :mod:`skfb.estimators` module implements fallback meta-estimators."""

__all__ = (
    "RateFallbackClassifier",
    "ThresholdFallbackClassifier",
    "ThresholdFallbackClassifierCV",
)

from ._threshold import (
    RateFallbackClassifier,
    ThresholdFallbackClassifier,
    ThresholdFallbackClassifierCV,
)
