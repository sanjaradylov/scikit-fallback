"""Tests classifiers w/ rejection option based on certainty thresholds."""

from sklearn.linear_model import LogisticRegression

import numpy as np
import pytest

from skfb.estimators import (
    predict_or_fallback,
    CoverageFallbackClassifierCV,
    RateFallbackClassifierCV,
    ThresholdFallbackClassifier,
    ThresholdFallbackClassifierCV,
    UtilityFallbackClassifierCV,
)

from .test_common import TestFallbackEstimator


@pytest.mark.parametrize(
    "y_score, y_true, fb_mask, threshold, ambiguity_threshold",
    [
        (
            np.array([[0.1, 0.2, 0.7], [0.33, 0.33, 0.34], [0.0, 1.0, 0.0]]),
            np.array([-1, -1, 1]),
            np.array([True, True, False]),
            0.8,
            0.05,
        ),
        (
            np.array([[0.099, 0.4, 0.501], [0.495, 0.4, 0.105], [0.4, 0.5, 0.1]]),
            np.array([2, -1, -1]),
            np.array([False, True, True]),
            0.2,
            0.1,
        ),
    ],
)
def test_predict_or_fallback(y_score, y_true, fb_mask, threshold, ambiguity_threshold):
    """Tests ``predict_or_fallback``."""
    rejector = TestFallbackEstimator()
    rejector.fit()
    classes = np.array([0, 1, 2])

    y_pred = predict_or_fallback(
        rejector,
        y_score,
        classes,
        threshold=threshold,
        ambiguity_threshold=ambiguity_threshold,
    )
    np.testing.assert_array_equal(y_true, y_pred)

    y_pred = predict_or_fallback(
        rejector,
        y_score,
        classes,
        threshold=threshold,
        ambiguity_threshold=ambiguity_threshold,
        fallback_mode="ignore",
    )
    assert -1 not in y_pred

    y_pred = predict_or_fallback(
        rejector,
        y_score,
        classes,
        threshold=threshold,
        ambiguity_threshold=ambiguity_threshold,
        fallback_mode="store",
    )
    np.testing.assert_array_equal(y_pred.get_dense_fallback_mask(), fb_mask)


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

    y_store = rejector.predict(X)
    np.testing.assert_array_equal(
        y_store.get_dense_fallback_mask(),
        np.array([False, False, False, False, True, True]),
    )
    np.testing.assert_array_equal(
        rejector.set_params(fallback_mode="return").predict(X),
        y_comb,
    )
    np.testing.assert_array_equal(
        y_store,
        rejector.set_params(fallback_mode="ignore").predict(X),
    )

    rejector.set_params(fallback_mode="store", threshold=0.5, ambiguity_threshold=0.2)
    np.testing.assert_array_equal(
        rejector.predict_proba(X).get_dense_fallback_mask(),
        np.array([False, False, False, False, True, True]),
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

    y_store = rejector.predict(X)
    np.testing.assert_array_equal(
        y_store.get_dense_fallback_mask(),
        np.array([False, False, False, False, True, False]),
    )
    np.testing.assert_array_equal(
        rejector.set_params(fallback_mode="return").predict(X),
        y_comb,
    )
    np.testing.assert_array_equal(
        y_store,
        rejector.set_params(fallback_mode="ignore").predict(X),
    )

    rejector.threshold_ = 0.5  # Of course, don't do this in prod...
    rejector.set_params(fallback_mode="store", ambiguity_threshold=0.2)
    np.testing.assert_array_equal(
        rejector.predict_proba(X).get_dense_fallback_mask(),
        np.array([False, False, False, False, True, True]),
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
def test_coverage_fallback_classifier(y_true, y_comb, fallback_label):
    """Tests fit, predict, and attrs of ``CoverageFallbackClassifierCV``."""
    X_accept = [[0, 0], [6, 6], [0, 1], [5, 6], [1, 1], [5, 5], [1, 0], [6, 5]]
    X_ambiguous = [[3.25, 3], [3.0, 3.25]]
    X = np.array(X_accept + X_ambiguous)

    estimator = LogisticRegression(random_state=0)
    coverage = 0.8
    cv = 2

    rejector = CoverageFallbackClassifierCV(estimator, coverage=coverage, cv=cv)
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


def test_rate_fallback_classifier_cv():
    """Tests that ``RateFallbackClassifierCV`` produces same results as Coverage."""
    X_accept = [[0, 0], [6, 6], [0, 1], [5, 6], [1, 1], [5, 5], [1, 0], [6, 5]]
    X_ambiguous = [[3.25, 3], [3.0, 3.25]]
    X = np.array(X_accept + X_ambiguous)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    estimator = LogisticRegression(random_state=0)

    rejector = RateFallbackClassifierCV(estimator, fallback_rate=0.2, cv=2)
    rejector.fit(X, y)
    assert hasattr(rejector, "threshold_")

    rejector_new = CoverageFallbackClassifierCV(estimator, coverage=0.8, cv=2)
    rejector_new.fit(X, y)
    np.testing.assert_almost_equal(rejector.threshold_, rejector_new.threshold_)


@pytest.mark.parametrize(
    "y_true, fallback_label",
    [
        (np.array([0, 1, 0, 1, 0, 1]), -1),
        (np.array([0, 1, 0, 1, 0, 1]), 2),
        (np.array(["a", "b", "a", "b", "a", "b"]), "c"),
    ],
)
def test_utility_fallback_classifier_cv(y_true, fallback_label):
    """Tests fit, predict, and attrs of ``UtilityFallbackClassifierCV``."""
    X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2.0, 2.5]])

    estimator = LogisticRegression(random_state=0)
    thresholds = (0.5, 0.55, 0.6, 0.65, 0.7)
    cv = 2

    rejector = UtilityFallbackClassifierCV(
        estimator,
        min_utility=0.8,
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
        "best_utility_",
    ]:
        assert hasattr(rejector, key)

    np.testing.assert_equal(
        rejector.fallback_label_.dtype,
        rejector.classes_.dtype,
    )

    # The learned threshold must achieve at least min_utility
    assert rejector.best_utility_ >= 0.8

    # Threshold should be the lowest feasible one
    mean_scores = rejector.cv_scores_.mean(axis=1)
    feasible = mean_scores >= 0.8
    if np.any(feasible):
        expected_threshold = np.asarray(thresholds)[feasible].min()
        assert rejector.threshold_ == expected_threshold

    y_pred = rejector.predict(X)
    assert hasattr(y_pred, "get_dense_fallback_mask")

    # Test with fallback_mode="return"
    y_return = rejector.set_params(fallback_mode="return").predict(X)
    assert y_return.dtype == rejector.classes_.dtype


def test_utility_fallback_classifier_cv_warning():
    """Tests that a warning is issued when no threshold meets min_utility."""
    X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2.0, 2.5]])
    y = np.array([0, 1, 0, 1, 0, 1])

    estimator = LogisticRegression(random_state=0)

    rejector = UtilityFallbackClassifierCV(
        estimator,
        min_utility=1.0,  # Unreachable utility
        thresholds=(0.5, 0.6, 0.7),
        cv=2,
    )

    with pytest.warns(UserWarning, match="No threshold achieves min_utility"):
        rejector.fit(X, y)

    # Should still have a valid threshold (falls back to best utility)
    assert hasattr(rejector, "threshold_")
