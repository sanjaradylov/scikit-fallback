"""The :mod:`skfb.ensemble` module implements cascade ensembles."""

__all__ = (
    "CascadeNotFittedWarning",
    "RoutingClassifier",
    "ThresholdCascadeClassifier",
    "ThresholdCascadeClassifierCV",
)

from ._routing import RoutingClassifier
from ._threshold import (
    CascadeNotFittedWarning,
    ThresholdCascadeClassifier,
    ThresholdCascadeClassifierCV,
)
