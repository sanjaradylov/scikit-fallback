"""Tests classifiers w/ rejection option based on local certainty thresholds."""

from sklearn.linear_model import LogisticRegression

import numpy as np
import pytest

from skfb.estimators import (
    multi_threshold_predict_or_fallback,
    MultiThresholdFallbackClassifier,
)

from .test_common import TestFallbackEstimator


@pytest.mark.parametrize(
    "y_score, y_true, thresholds, ambiguity_threshold",
    [
        (
            np.array(
                [
                    [0.33, 0.33, 0.34],
                    [0.11, 0.79, 0.1],
                    [0.89, 0.1, 0.01],
                    [0.29, 0.33, 0.36],
                    [0.09, 0.81, 0.1],
                    [0.91, 0.08, 0.01],
                ],
            ),
            np.array([-1, -1, -1, 2, 1, 0]),
            {0: 0.9, 1: 0.8, 2: 0.35},
            0.0,
        ),
        (
            np.array(
                [
                    [0.33, 0.33, 0.34],
                    [0.11, 0.79, 0.1],
                    [0.89, 0.1, 0.01],
                    [0.29, 0.33, 0.36],
                    [0.09, 0.81, 0.1],
                    [0.91, 0.08, 0.01],
                ],
            ),
            np.array([-1, -1, -1, -1, 1, 0]),
            {0: 0.9, 1: 0.8, 2: 0.35},
            0.1,
        ),
    ],
)
def test_predict_or_fallback(y_score, y_true, thresholds, ambiguity_threshold):
    """Tests ``multi_threshold_predict_or_fallback``."""
    rejector = TestFallbackEstimator()
    rejector.fit()
    classes = np.array([0, 1, 2])

    y_pred = multi_threshold_predict_or_fallback(
        rejector,
        y_score,
        classes=classes,
        thresholds=thresholds,
        ambiguity_threshold=ambiguity_threshold,
    )
    np.testing.assert_array_equal(y_true, y_pred)


def test_multi_threshold_fallback_classifier():
    """Tests fit, predict, and attrs of ``MultiThresholdFallbackClassifier``."""
    X = np.array(
        [
            [-3, -3],
            [-3, -2],
            [-2, -3],
            [2, 3],
            [2, 4],
            [3, 4],
            [3.1, 3],
            [3, 2.9],
            [4, 3],
            [3, 2],
            [4, 2],
            [2.9, 3],
            [3, 3.1],
        ]
    )
    y = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 5)
    y_comb = np.array(["a"] * 3 + ["b"] * 3 + ["d"] * 2 + ["c"] * 3 + ["d"] * 2)

    estimator = LogisticRegression(C=10_000, random_state=0)
    thresholds = {"a": 0.99, "b": 0.8, "c": 0.75}
    rejector = MultiThresholdFallbackClassifier(
        estimator,
        thresholds=thresholds,
        fallback_label="d",
    )
    rejector.fit(X, y)

    for key in ["classes_", "fallback_label_"]:
        assert hasattr(rejector, key)

    np.testing.assert_equal(
        rejector.fallback_label_.dtype,
        rejector.classes_.dtype,
    )
    np.testing.assert_array_equal(
        rejector.predict(X).get_dense_fallback_mask(),
        np.array([False] * 3 + [False] * 3 + [True] * 2 + [False] * 3 + [True] * 2),
    )
    np.testing.assert_array_equal(
        rejector.set_params(fallback_mode="return").predict(X),
        y_comb,
    )
