"""Tests classification metrics w/ a rejection option."""

import numpy as np
import pytest

from skfb.core import array as ska
from skfb.metrics import predict_accept_confusion_matrix, predict_reject_accuracy_score


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
