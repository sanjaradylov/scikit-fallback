"""The :mod:`skfb.ensemble` module implements cascade ensembles."""

__all__ = (
    "CascadeNotFittedWarning",
    "SingleModelRouterClassifier",
    "ThresholdCascadeClassifier",
    "ThresholdCascadeClassifierCV",
)

from ._router import SingleModelRouterClassifier
from ._threshold import (
    CascadeNotFittedWarning,
    ThresholdCascadeClassifier,
    ThresholdCascadeClassifierCV,
)
