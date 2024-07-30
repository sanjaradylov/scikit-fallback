"""Enables :class:`~skfb.estimators.MultiThresholdFallbackClassifierCV`.

    >>> from skfb.experimental import enable_multi_threshold_fallback_classifier_cv
    >>> from skfb.estimators import MultiThresholdFallbackClassifierCV
"""

from .. import estimators
from ..estimators._multi_threshold import MultiThresholdFallbackClassifierCV


setattr(
    estimators,
    "MultiThresholdFallbackClassifierCV",
    MultiThresholdFallbackClassifierCV,
)
estimators.__all__ += ("MultiThresholdFallbackClassifierCV",)
