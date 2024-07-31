"""Tests classification metrics w/ a rejection option."""

import numpy as np
import pytest

from sklearn.metrics import log_loss

from skfb.core import array as ska

# pylint: disable=unused-import
from skfb.experimental import enable_error_rejection_loss
from skfb.metrics import (
    error_rejection_loss,
    predict_accept_confusion_matrix,
    predict_reject_accuracy_score,
    predict_reject_recall_score,
)


@pytest.mark.parametrize(
    "y_true, y_pred, cm_true",
    [
        # region Arbitrary binary predictions and rejections
        (
            np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([[1, 2], [3, 4]]),
        ),
        # endregion
        # region All-rejects
        (
            np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([[3, 0], [7, 0]]),
        ),
        # endregion
        # region All-accepts
        (
            np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([[0, 3], [0, 7]]),
        ),
        # endregion
        # region Arbitrary multiclass predictions and rejections
        (
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
            ska.fbarray([0, 1, 2, 1, 2, 0, 1, 2, 0, 2], [0, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
            np.array([[4, 3], [2, 1]]),
        ),
        (
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.str_),
            ska.fbarray(
                np.array([0, 1, 2, 1, 2, 0, 1, 2, 0, 2], dtype=np.str_),
                [0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            ),
            np.array([[4, 3], [2, 1]]),
        ),
        # endregion
        # region Empty predictions and rejections
        ([], ska.fbarray([]), np.ndarray((0, 0), dtype=np.int64)),
        # endregion
    ],
)
def test_successful_predict_accept_confusion_matrix(y_true, y_pred, cm_true):
    cm_pred = predict_accept_confusion_matrix(y_true, y_pred)
    np.testing.assert_array_equal(cm_true, cm_pred)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        # region Different lengths
        ([], [1]),
        ([1], []),
        # endregion
        # region Invalid type of ``y_pred``
        ([], []),
        ([1, 0, 1], [0, 0, 1]),
        # endregion
    ],
)
def test_failed_predict_accept_confusion_matrix(y_true, y_pred):
    with pytest.raises(ValueError):
        predict_accept_confusion_matrix(y_true, y_pred)


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (
            np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            (4 + 1) / (4 + 1 + 2 + 3),
        ),
        (
            np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            (7 + 0) / (7 + 0 + 0 + 3),
        ),
        (
            np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            (0 + 3) / (0 + 3 + 0 + 7),
        ),
        (
            np.array([0, 0, 0, 0, 0]),
            ska.fbarray([0, 1, 0, 1, 0], [1, 0, 1, 0, 1]),
            (0 + 0) / (0 + 0 + 2 + 3),
        ),
        (
            [0],
            ska.fbarray([1], [0]),
            (0 + 0) / (0 + 0 + 1 + 0),
        ),
        (
            [0],
            ska.fbarray([1], [1]),
            (0 + 1) / (0 + 1 + 0 + 0),
        ),
    ],
)
def test_predict_reject_accuracy_score(y_true, y_pred, result):
    assert predict_reject_accuracy_score(y_true, y_pred) == result


@pytest.mark.parametrize(
    "y_true, y_pred, beta, result",
    [
        (
            np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            0.4,
            4 / (4 + 3) * 0.4 + 1 / (1 + 2) * 0.6,
        ),
        (
            np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            0.1,
            7 / (7 + 0) * 0.1,
        ),
        (
            np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]),
            ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            0.1,
            3 / (3 + 0) * (1 - 0.1),
        ),
    ],
)
def test_predict_reject_recall_score(y_true, y_pred, beta, result):
    assert predict_reject_recall_score(y_true, y_pred, beta=beta) == result


@pytest.mark.parametrize(
    "y_true, y_prob, thresholds, y_pred, class_weight, score_func, loss",
    [
        (
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array(
                [
                    [0.4, 0.3, 0.3],  # Class_0 - fallback
                    [0.5, 0.3, 0.2],  # Class_0 - accept
                    [0.4, 0.5, 0.1],  # Class_1 - fallback
                    [0.3, 0.6, 0.1],  # Class_1 - fallback
                    [0.2, 0.7, 0.1],  # Class_1 - accept
                    [0.2, 0.2, 0.6],  # Class_2 - fallback
                    [0.1, 0.2, 0.7],  # Class_2 - fallback
                    [0.1, 0.1, 0.8],  # Class_2 - fallback
                    [0.1, 0.0, 0.9],  # Class_2 - accept
                ],
            ),
            [0.41, 0.61, 0.81],
            None,  #  Converted to np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
            None,
            None,
            1 / 4 * (1 / 2 + 2 / 3 + 3 / 4) + 1 / 4 * 2 / 9,
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array(
                [
                    [0.4, 0.3, 0.3],  # Class_0 - fallback
                    [0.5, 0.3, 0.2],  # Class_0 - accept
                    [0.4, 0.5, 0.1],  # Class_1 - fallback
                    [0.3, 0.6, 0.1],  # Class_1 - fallback
                    [0.2, 0.7, 0.1],  # Class_1 - accept
                    [0.2, 0.2, 0.6],  # Class_2 - fallback
                    [0.1, 0.2, 0.7],  # Class_2 - fallback
                    [0.1, 0.1, 0.8],  # Class_2 - fallback
                    [0.1, 0.0, 0.9],  # Class_2 - accept
                ],
            ),
            [0.41, 0.61, 0.81],
            None,  #  Converted to np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
            [1 / 3, 0, 1 / 2, 1 / 6],
            None,
            1 / 3 * 1 / 2 + 0 * 2 / 3 + 1 / 2 * 3 / 4 + 1 / 6 * 2 / 9,
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array(
                [
                    [0.4, 0.3, 0.3],  # Class_0 - fallback
                    [0.5, 0.3, 0.2],  # Class_0 - accept
                    [0.4, 0.5, 0.1],  # Class_1 - fallback
                    [0.3, 0.6, 0.1],  # Class_1 - fallback
                    [0.2, 0.7, 0.1],  # Class_1 - accept
                    [0.2, 0.2, 0.6],  # Class_2 - fallback
                    [0.1, 0.2, 0.7],  # Class_2 - fallback
                    [0.1, 0.1, 0.8],  # Class_2 - fallback
                    [0.1, 0.0, 0.9],  # Class_2 - accept
                ],
            ),
            [0.41, 0.61, 0.81],
            np.array([0, 1, 3, 1, 3, 3, 2, 2, 3]),
            None,
            None,
            1 / 4 * (1 / 2 + 2 / 3 + 3 / 4) + 1 / 4 * 5 / 9,
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array(
                [
                    [0.4, 0.3, 0.3],  # Class_0 - fallback
                    [0.5, 0.3, 0.2],  # Class_0 - accept
                    [0.4, 0.5, 0.1],  # Class_1 - fallback
                    [0.3, 0.6, 0.1],  # Class_1 - fallback
                    [0.2, 0.7, 0.1],  # Class_1 - accept
                    [0.2, 0.2, 0.6],  # Class_2 - fallback
                    [0.1, 0.2, 0.7],  # Class_2 - fallback
                    [0.1, 0.1, 0.8],  # Class_2 - fallback
                    [0.1, 0.0, 0.9],  # Class_2 - accept
                ],
            ),
            [0.41, 0.61, 0.81],
            np.array([0, 1, 3, 1, 3, 3, 2, 2, 3]),
            [1 / 4, 1 / 16, 3 / 16, 1 / 2],
            None,
            1 / 4 * 1 / 2 + 1 / 16 * 2 / 3 + 3 / 16 * 3 / 4 + 1 / 2 * 5 / 9,
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array(
                [
                    [0.4, 0.3, 0.3],  # Class_0 - fallback
                    [0.5, 0.3, 0.2],  # Class_0 - accept
                    [0.4, 0.5, 0.1],  # Class_1 - fallback
                    [0.3, 0.6, 0.1],  # Class_1 - fallback
                    [0.2, 0.7, 0.1],  # Class_1 - accept
                    [0.2, 0.2, 0.6],  # Class_2 - fallback
                    [0.1, 0.2, 0.7],  # Class_2 - fallback
                    [0.1, 0.1, 0.8],  # Class_2 - fallback
                    [0.1, 0.0, 0.9],  # Class_2 - accept
                ],
            ),
            [0.41, 0.61, 0.81],
            None,
            [1 / 4, 1 / 16, 3 / 16, 1 / 2],
            log_loss,
            1 / 4 * 1 / 2 + 1 / 16 * 2 / 3 + 3 / 16 * 3 / 4 + 1 / 2 * 0.632,
        ),
    ],
)
def test_error_rejection_loss(
    y_true, y_prob, thresholds, y_pred, class_weight, score_func, loss
):
    np.testing.assert_almost_equal(
        loss,
        error_rejection_loss(
            y_true,
            y_prob,
            thresholds=thresholds,
            y_pred=y_pred,
            score_func=score_func,
            class_weight=class_weight,
        ),
        decimal=3,
    )
