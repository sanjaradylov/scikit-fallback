"""Tests threshold-based cascade ensembles."""

import time

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from skfb.ensemble import ThresholdCascadeClassifier


class OverheadEstimator(BaseEstimator, ClassifierMixin):
    """Accepts and returns the same probability scores."""

    def __init__(self, overhead):
        self.overhead = overhead

    # pylint: disable=unused-argument
    def fit(self, X, y):
        return self

    def predict(self, y):
        time.sleep(self.overhead * len(y))
        return y.argmax(axis=1)

    def predict_proba(self, y):
        time.sleep(self.overhead * len(y))
        return y


def test_threshold_cascade_classifier():
    """Tests ``TresholdCascadeClassifier``."""
    y_score = np.array(
        [
            [0.8, 0.2],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.49, 0.51],
            [0.71, 0.29],
            [0.39, 0.61],
            [0.55, 0.45],
            [0.35, 0.65],
            [0.505, 0.495],
            [0.39, 0.61],
            [0.5, 0.5],
            [0.51, 0.49],
        ],
    )
    y = np.array(list("acacaccacaac"))
    n_iterations = 100
    X = np.concat([y_score] * n_iterations)
    y = np.concat([y] * n_iterations)

    weak = OverheadEstimator(overhead=1e-4)
    medium = OverheadEstimator(overhead=5e-4)
    large = OverheadEstimator(overhead=1e-3)
    cascade = ThresholdCascadeClassifier(
        [weak, medium, large],
        [0.7, 0.6],
        return_earray=True,
    )
    cascade.fit(X, y)

    tic = time.perf_counter()
    cascade.predict(X)
    toc = time.perf_counter()
    cascade_time = toc - tic

    tic = time.perf_counter()
    cascade.set_estimators(0).predict(X)
    toc = time.perf_counter()
    weak_time = toc - tic

    tic = time.perf_counter()
    cascade.set_estimators(2).predict(X)
    toc = time.perf_counter()
    large_time = toc - tic

    assert large_time > cascade_time > weak_time

    y_pred = cascade.reset_estimators().predict(y_score)
    np.testing.assert_array_equal(
        y_pred.ensemble_mask.toarray(),
        np.array(
            [
                [True, False, False],
                [False, True, False],
                [True, False, False],
                [False, False, True],
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [False, True, False],
                [False, False, True],
                [False, True, False],
                [False, False, True],
                [False, False, True],
            ],
        ),
    )

    np.testing.assert_array_equal(
        y_pred.acceptance_rates,
        np.array([3 / 12, 4 / 12, 5 / 12]),
    )
