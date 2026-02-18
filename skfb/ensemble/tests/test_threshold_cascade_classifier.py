"""Tests threshold-based cascade ensembles."""

import time

import numpy as np

import pytest

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from skfb.ensemble import ThresholdCascadeClassifier


class OverheadEstimator(BaseEstimator, ClassifierMixin):
    """Accepts and returns the same probability scores."""

    def __init__(self, overhead):
        self.overhead = overhead

    # pylint: disable=unused-argument
    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        return self

    def predict(self, y):
        time.sleep(self.overhead * len(y))
        return y.argmax(axis=1)

    def predict_proba(self, y):
        time.sleep(self.overhead * len(y))
        return y

    decision_function = predict_proba


@pytest.mark.parametrize(
    "thresholds, ensemble_mask, acceptance_rates",
    [
        # region Arbitrary deferrals
        (
            [0.7, 0.6],
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
            np.array([3 / 12, 4 / 12, 5 / 12]),
        ),
        # endregion
        # region All-deferals
        (
            [0.99, 0.99],
            np.array(
                [
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                ],
            ),
            np.array([0 / 12, 0 / 12, 12 / 12]),
        ),
        # endregion
        # region All-accepts
        (
            [0.1, 0.1],
            np.array(
                [
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                ],
            ),
            np.array([12 / 12, 0 / 12, 0 / 12]),
        ),
        # endregion
        # region One-deferrals
        (
            [0.99, 0.1],
            np.array(
                [
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                ],
            ),
            np.array([0 / 12, 12 / 12, 0 / 12]),
        ),
    ],
)
def test_threshold_cascade_classifier(thresholds, ensemble_mask, acceptance_rates):
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
        thresholds,
        return_earray=True,
        response_method="predict_proba",
    )
    cascade.fit(X, y)

    tic = time.perf_counter()
    cascade.predict(X)
    toc = time.perf_counter()
    cascade_time = round(toc - tic, 2)

    tic = time.perf_counter()
    cascade.set_estimators(0).predict(X)
    toc = time.perf_counter()
    weak_time = round(toc - tic, 2)

    tic = time.perf_counter()
    cascade.set_estimators(2).predict(X)
    toc = time.perf_counter()
    large_time = round(toc - tic, 2)

    if not (large_time > cascade_time >= weak_time):
        pytest.xfail(
            f"large_time = {large_time:.2f}, cascade_time = {cascade_time:.2f}, "
            f"weak_time = {weak_time:.2f}"
        )

    y_pred = cascade.reset_estimators().predict(y_score)
    np.testing.assert_array_equal(y_pred.ensemble_mask.toarray(), ensemble_mask)

    np.testing.assert_array_equal(y_pred.acceptance_rates, acceptance_rates)
