"""scikit-fallback: machine learning with a reject option."""

import sys
import warnings


if not sys.warnoptions:
    warnings.simplefilter(action="default")


# pylint: disable=wrong-import-position
from . import (
    core,
    estimators,
    metrics,
)

__all__ = (
    "core",
    "estimators",
    "metrics",
)

__version__ = "0.1.1.post0"
