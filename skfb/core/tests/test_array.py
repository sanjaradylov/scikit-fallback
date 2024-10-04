"""Tests array objects."""

import numpy as np
import pytest

from skfb.core import array as ska


@pytest.mark.parametrize(
    "predictions, fallback_mask",
    [
        (
            [0, 1, 0, 1],
            [0, 1, 0, 0],
        ),
        (
            [0, 1, 0, 1],
            [0, 0, 0, 0],
        ),
        (
            [0, 1, 0, 1],
            [1, 1, 1, 1],
        ),
        (
            (0, 1, 0, 1),
            [0.0, 0.0, 0.0, 0.0],
        ),
        (
            np.array([0, 1, 0, 1]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ),
        (
            np.array([0, 1, 0, 1]),
            (1.0, 1.0, 1.0, 1.0),
        ),
        (
            [0] * 5 + [1] * 5,
            [],
        ),
        (
            [],
            [],
        ),
        (
            list("010212"),
            (False, True, False, True, False, False),
        ),
    ],
)
def test_successful_fbndarray(predictions, fallback_mask):
    y_pred = ska.fbarray(predictions, fallback_mask)

    np.testing.assert_array_equal(
        y_pred.get_dense_fallback_mask(),
        np.asarray(fallback_mask, dtype=np.bool_),
    )

    y_comb = np.asarray(predictions).copy()
    fallback_mask = np.asarray(fallback_mask, dtype=np.bool_)
    fallback_label = np.asarray(3, dtype=y_pred.dtype)
    y_comb[fallback_mask] = fallback_label

    np.testing.assert_array_equal(
        y_pred.as_comb(fallback_label=fallback_label),
        y_comb,
    )

    assert (y_pred == fallback_label).sum() == 0


@pytest.mark.parametrize(
    "predictions, fallback_mask",
    [
        (
            [0, 1, 0, 1],
            [0, 0, 1],
        ),
        (
            [],
            [1, 0, 1, 1],
        ),
    ],
)
def test_failed_fbndarray(predictions, fallback_mask):
    with pytest.raises(ValueError):
        ska.fbarray(predictions, fallback_mask)
