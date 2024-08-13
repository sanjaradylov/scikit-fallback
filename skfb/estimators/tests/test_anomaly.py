"""Tests classifiers with a reject option based on anomaly detection."""

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from skfb.estimators import AnomalyFallbackClassifier


def test_anomaly_fallback_classifier():
    """Tests features of ``skfb.estimators.AnomalyFallbackClassifier``."""
    X = np.array(
        [
            [0, 0],
            [10, 10],
            [1, 1],
            [9, 9],
            [1, 0],
            [9, 10],
            [0, 1],
            [10, 9],
            [5.5, 5],
            [5.0, 5.5],
        ]
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    estimator = LogisticRegression(random_state=0)
    outlier_detector = IsolationForest(
        n_estimators=10, max_samples=1.0, contamination=0.2, random_state=0
    )
    rejector = AnomalyFallbackClassifier(estimator, outlier_detector)
    rejector.fit(X, y)

    for key in ["classes_", "fallback_label_", "outlier_detector_", "estimator_"]:
        assert hasattr(rejector, key)

    true_fallback_mask = np.array([False] * 8 + [True] * 2)

    np.testing.assert_array_equal(
        true_fallback_mask,
        rejector.predict(X).get_dense_fallback_mask(),
    )

    np.testing.assert_array_equal(
        np.array([0, 1, 0, 1, 0, 1, 0, 1, -1, -1]),
        rejector.set_params(fallback_mode="return").predict(X),
    )

    y_prob = rejector.predict_proba(X)
    assert y_prob.shape == (len(X), 2)
    np.testing.assert_array_equal(
        true_fallback_mask,
        y_prob.get_dense_fallback_mask(),
    )

    y_score = rejector.decision_function(X)
    assert y_score.shape == (len(X),)
    np.testing.assert_array_equal(
        true_fallback_mask,
        y_score.get_dense_fallback_mask(),
    )
