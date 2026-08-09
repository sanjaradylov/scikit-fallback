"""Tests common metrics."""

import numpy as np
import pytest

from sklearn.metrics import accuracy_score, f1_score

from skfb.metrics import prediction_quality, utility_coverage_curve


@pytest.mark.parametrize(
    "score_func, result",
    [
        (accuracy_score, 0.75),
        (f1_score, 0.66),
    ],
)
def test_nonempty_prediction_quality_score(score_func, result):
    """Tests prediction quality on non-rejected examples."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0, -1, -1])

    assert (
        pytest.approx(
            prediction_quality(
                y_true,
                y_pred,
                score_func,
                fallback_label=-1,
            ),
            rel=1e-2,
        )
        == result
    )


@pytest.mark.parametrize(
    "sample_weight, true_utility",
    [
        (None, np.array([1 / 2, 3 / 4, 3 / 5])),
        (np.ones(5), np.array([1 / 2, 3 / 4, 3 / 5])),
        (
            np.arange(1, 6),
            np.array(
                [
                    1 * 1 / (1 * 1 + 1 * 2),
                    (1 * 1 + 1 * 3 + 1 * 4) / (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4),
                    (1 * 1 + 1 * 3 + 1 * 4) / (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5),
                ],
            ),
        ),
    ],
)
def test_utility_coverage_curve(sample_weight, true_utility):
    """Sample weights are propagated while evaluating each threshold prefix."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 1, 0, 0])
    y_score = np.array([0.9, 0.9, 0.7, 0.7, 0.1])

    utility, coverage, thresholds = utility_coverage_curve(
        y_true,
        y_pred,
        y_score=y_score,
        scoring=accuracy_score,
        sample_weight=sample_weight,
    )

    np.testing.assert_allclose(utility, true_utility)
    np.testing.assert_allclose(coverage, np.array([0.4, 0.8, 1.0]))
    np.testing.assert_allclose(thresholds, np.array([0.9, 0.7, 0.1]))


def test_utility_coverage_curve_infer_from_2d():
    """When y_score is None and y_pred is 2D, scores and labels are inferred."""
    y_true = np.array([0, 1, 0, 2, 2, 1, 0, 0, 1, 0])
    y_proba = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.40, 0.35, 0.25],
            [0.90, 0.05, 0.05],
            [0.05, 0.85, 0.10],
            [0.10, 0.10, 0.80],
            [0.20, 0.60, 0.20],
            [0.20, 0.20, 0.60],
            [0.70, 0.20, 0.10],
            [0.70, 0.20, 0.10],
            [0.55, 0.25, 0.20],
        ]
    )

    # Explicit y_score path
    y_pred = np.argmax(y_proba, axis=1)
    y_score = np.max(y_proba, axis=1)
    u_explicit, c_explicit, t_explicit = utility_coverage_curve(
        y_true, y_pred, y_score=y_score
    )

    # Inferred path (2D y_pred, no y_score)
    u_inferred, c_inferred, t_inferred = utility_coverage_curve(y_true, y_proba)

    np.testing.assert_allclose(u_inferred, u_explicit)
    np.testing.assert_allclose(c_inferred, c_explicit)
    np.testing.assert_allclose(t_inferred, t_explicit)


def test_utility_coverage_curve_infer_with_string_labels():
    """String labels are mapped correctly via the labels parameter."""
    y_true = np.array(["cat", "dog", "cat", "bird"])
    y_proba = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    labels = np.array(["cat", "dog", "bird"])

    utility, coverage, thresholds = utility_coverage_curve(
        y_true, y_proba, labels=labels
    )

    # Scores: [0.9, 0.8, 0.7, 0.8] → sorted: [0.9, 0.8, 0.8, 0.7]
    # Unique thresholds: 0.9 (end idx 0), 0.8 (end idx 2), 0.7 (end idx 3)
    np.testing.assert_allclose(thresholds, np.array([0.9, 0.8, 0.7]))
    np.testing.assert_allclose(coverage, np.array([0.25, 0.75, 1.0]))
    # All predictions match y_true at every coverage level
    np.testing.assert_allclose(utility, np.array([1.0, 1.0, 1.0]))


def test_utility_coverage_curve_raises_without_score_and_1d():
    """ValueError raised when y_pred is 1D and y_score is not given."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 1])

    with pytest.raises(ValueError, match="y_score must be provided"):
        utility_coverage_curve(y_true, y_pred)


def test_utility_coverage_curve_clips_coverage_range():
    """Coverage levels outside [min_coverage, max_coverage] are dropped."""
    y_true = np.arange(10) % 2
    y_pred = y_true.copy()
    y_score = np.linspace(0.1, 1.0, 10)

    utility, coverage, thresholds = utility_coverage_curve(
        y_true,
        y_pred,
        y_score=y_score,
        min_coverage=0.3,
        max_coverage=0.7,
    )

    np.testing.assert_allclose(coverage, np.array([0.3, 0.4, 0.5, 0.6, 0.7]))
    assert len(utility) == len(thresholds) == len(coverage)


def test_utility_coverage_curve_drops_low_coverage_tail_by_default():
    """The default min_coverage discards the high-variance low-coverage tail."""
    y_true = np.arange(100) % 2
    y_pred = y_true.copy()
    y_score = np.linspace(0.01, 1.0, 100)

    _, coverage, _ = utility_coverage_curve(y_true, y_pred, y_score=y_score)

    assert coverage.min() >= 0.05
    np.testing.assert_allclose(coverage.max(), 1.0)


@pytest.mark.parametrize(
    "min_coverage, max_coverage, match",
    [
        (0.8, 0.2, "min_coverage should be less than max_coverage"),
        (0.0, 0.05, "No coverage level falls within"),
    ],
)
def test_utility_coverage_curve_invalid_range(min_coverage, max_coverage, match):
    """Inverted or empty coverage ranges raise informative errors."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])

    with pytest.raises(ValueError, match=match):
        utility_coverage_curve(
            y_true,
            y_pred,
            y_score=y_score,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
        )
