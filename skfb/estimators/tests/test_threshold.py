"""Tests classifiers w/ rejection option based on certainty thresholds."""

from sklearn.linear_model import LogisticRegression

import numpy as np
import pytest

from skfb.estimators import (
    RateFallbackClassifier,
    ThresholdFallbackClassifier,
    ThresholdFallbackClassifierCV,
)


@pytest.mark.parametrize(
    "y_true, y_comb, fallback_label",
    [
        (
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([0, 1, 0, 1, -1, -1]),
            -1,
        ),
        (
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([0, 1, 0, 1, 2, 2]),
            2,
        ),
        (
            np.array(["a", "b", "a", "b", "a", "b"]),
            np.array(["a", "b", "a", "b", "c", "c"]),
            "c",
        ),
    ],
)
def test_threshold_fallback_classifier(y_true, y_comb, fallback_label):
    """Tests fit, predict, and attrs of ``ThresholdFallbackClassifier``.

    Only two last examples will be rejected by the estimator.
    """
    X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2.0, 2.5]])

    estimator = LogisticRegression(random_state=0)
    threshold = 0.6

    rejector = ThresholdFallbackClassifier(
        estimator,
        threshold=threshold,
        fallback_label=fallback_label,
        fallback_mode="store",
    )
    rejector.fit(X, y_true)

    for key in ["classes_", "fallback_label_"]:
        assert hasattr(rejector, key)

    np.testing.assert_equal(
        rejector.fallback_label_.dtype,
        rejector.classes_.dtype,
    )
    np.testing.assert_array_equal(
        rejector.predict(X).get_dense_fallback_mask(),
        np.array([False, False, False, False, True, True]),
    )
    np.testing.assert_array_equal(
        rejector.set_params(fallback_mode="return").predict(X),
        y_comb,
    )


@pytest.mark.parametrize(
    "y_true, y_comb, fallback_label",
    [
        (
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([0, 1, 0, 1, -1, 1]),
            -1,
        ),
        (
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([0, 1, 0, 1, 2, 1]),
            2,
        ),
        (
            np.array(["a", "b", "a", "b", "a", "b"]),
            np.array(["a", "b", "a", "b", "c", "b"]),
            "c",
        ),
    ],
)
def test_threshold_fallback_classifier_cv(y_true, y_comb, fallback_label):
    """Tests fit, predict, and attrs of ``ThresholdFallbackClassifierCV``.

    Only the second last will be rejected by the estimator.
    """
    X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2.0, 2.5]])

    estimator = LogisticRegression(random_state=0)
    thresholds = (0.5, 0.55, 0.6, 0.65)
    cv = 2

    rejector = ThresholdFallbackClassifierCV(
        estimator,
        thresholds=thresholds,
        cv=cv,
        fallback_label=fallback_label,
        fallback_mode="store",
    )
    rejector.fit(X, y_true)

    for key in [
        "classes_",
        "fallback_label_",
        "cv_",
        "cv_scores_",
        "scoring_",
        "threshold_",
        "best_score_",
    ]:
        assert hasattr(rejector, key)

    np.testing.assert_equal(
        rejector.fallback_label_.dtype,
        rejector.classes_.dtype,
    )
    np.testing.assert_array_equal(
        rejector.predict(X).get_dense_fallback_mask(),
        np.array([False, False, False, False, True, False]),
    )
    np.testing.assert_array_equal(
        rejector.set_params(fallback_mode="return").predict(X),
        y_comb,
    )


@pytest.mark.parametrize(
    "y_true, y_comb, fallback_label",
    [
        (
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, -1, -1]),
            -1,
        ),
        (
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 2, 2]),
            2,
        ),
        (
            np.array(["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]),
            np.array(["a", "b", "a", "b", "a", "b", "a", "b", "c", "c"]),
            "c",
        ),
    ],
)
def test_rate_fallback_classifier(y_true, y_comb, fallback_label):
    """Tests fit, predict, and attrs of ``RateFallbackClassifier``."""
    X_accept = [[0, 0], [6, 6], [0, 1], [5, 6], [1, 1], [5, 5], [1, 0], [6, 5]]
    X_ambiguous = [[3.25, 3], [3.0, 3.25]]
    X = np.array(X_accept + X_ambiguous)

    estimator = LogisticRegression(random_state=0)
    fallback_rate = 0.2
    cv = 2

    rejector = RateFallbackClassifier(estimator, fallback_rate=fallback_rate, cv=cv)
    rejector.set_params(fallback_label=fallback_label).fit(X, y_true)

    for key in ["cv_", "estimator_", "threshold_", "thresholds_"]:
        assert hasattr(rejector, key)

    np.testing.assert_equal(
        rejector.fallback_label_.dtype,
        rejector.classes_.dtype,
    )
    np.testing.assert_array_equal(
        rejector.predict(X).get_dense_fallback_mask(),
        np.array([False, False, False, False, False, False, False, False, True, True]),
    )
    np.testing.assert_array_equal(
        rejector.set_params(fallback_mode="return").predict(X),
        y_comb,
    )
