"""Tests classifiers w/ rejection option based on certainty thresholds."""

from sklearn.linear_model import LogisticRegression

import numpy as np
import pytest

from skfb.estimators import (
    predict_or_fallback,
    RateFallbackClassifierCV,
    ThresholdFallbackClassifier,
    ThresholdFallbackClassifierCV,
)


class TestFallbackEstimator:
    """Accepts and returns the same probability scores."""

    # pylint: disable=unused-argument
    def fit(self, X_unused=None, y_unused=None):
        self.is_fitted_ = True
        return self

    def predict_proba(self, y):
        return y


@pytest.mark.parametrize(
    "y_score, y_true, threshold, ambiguity_threshold",
    [
        (
            np.array([[0.1, 0.2, 0.7], [0.33, 0.33, 0.34], [0.0, 1.0, 0.0]]),
            np.array([-1, -1, 1]),
            0.8,
            0.05,
        ),
        (
            np.array([[0.099, 0.4, 0.501], [0.495, 0.4, 0.105], [0.4, 0.5, 0.1]]),
            np.array([2, -1, -1]),
            0.2,
            0.1,
        ),
    ],
)
def test_predict_or_fallback(y_score, y_true, threshold, ambiguity_threshold):
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
    """Tests fit, predict, and attrs of ``RateFallbackClassifierCV``."""
    X_accept = [[0, 0], [6, 6], [0, 1], [5, 6], [1, 1], [5, 5], [1, 0], [6, 5]]
    X_ambiguous = [[3.25, 3], [3.0, 3.25]]
    X = np.array(X_accept + X_ambiguous)

    estimator = LogisticRegression(random_state=0)
    fallback_rates = [0.2]
    cv = 2

    rejector = RateFallbackClassifierCV(estimator, fallback_rates=fallback_rates, cv=cv)
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
