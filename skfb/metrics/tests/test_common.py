"""Tests common metrics."""

import numpy as np
import pytest

from sklearn.metrics import accuracy_score, f1_score

from skfb.metrics import prediction_quality


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
