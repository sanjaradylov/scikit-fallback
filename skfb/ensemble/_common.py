"""Common utilities for cascades."""

import inspect

from sklearn.base import clone


def fit_one(estimator, X, y, sample_weight=None):
    """Trains ``estimator`` on ``(X, y)``."""
    estimator = clone(estimator)
    if "sample_weight" in inspect.getfullargspec(estimator.fit).args:
        return estimator.fit(X, y, sample_weight=sample_weight)
    else:
        return estimator.fit(X, y)
