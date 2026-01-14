"""Common utilities for cascades."""

import inspect

import numpy as np

from sklearn.base import clone


def fit_one(estimator, X, y, sample_weight=None):
    """Trains ``estimator`` on ``(X, y)``."""
    estimator = clone(estimator)
    if "sample_weight" in inspect.getfullargspec(estimator.fit).args:
        return estimator.fit(X, y, sample_weight=sample_weight)
    else:
        return estimator.fit(X, y)


def fit_and_predict_one(estimator, X, y, sample_weight=None):
    """Trains ``estimator`` on ``(X, y)`` and returns predictions."""
    estimator = fit_one(estimator, X, y, sample_weight=sample_weight)
    return estimator.predict(X)


def fit_and_predict_on_test(estimator, X_train, y_train, sample_weight, X_test):
    """Fit estimator and predict on test set."""
    fitted = fit_one(estimator, X_train, y_train, sample_weight)
    return np.asarray(fitted.predict(X_test))
