"""Classification metrics w/ a rejection option."""

__all__ = (
    "predict_accept_confusion_matrix",
    "predict_reject_accuracy_score",
)

import warnings

from sklearn.metrics import confusion_matrix

# pylint: disable=import-error,no-name-in-module
# pyright: reportMissingModuleSource=false
from sklearn.utils._param_validation import StrOptions, validate_params
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.utils.multiclass import type_of_target

import numpy as np

from ..core import array as ska


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": [ska.FBNDArray],
        "labels": ["array-like", None],
        "sample_weight": ["array-like", None],
        "normalize": [StrOptions({"true", "pred", "all"}), None],
    },
    prefer_skip_nested_validation=True,
)
def predict_accept_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    sample_weight=None,
    normalize=None,
):
    """Computes confusion matrix w/ rows as accuracy and columns as acceptance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by both a rejector and a classifier.

    labels : array-like of shape (2,), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    C : ndarray of shape (2, 2)
        TR (true-reject)   FA (false-accept)
        FR (false-reject)  TA (true-accept)

    See Also
    --------
    sklearn.metrics.confusion_matrix : True vs Predicted confusion matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.core import array as ska
    >>> from skfb.metrics import predict_accept_confusion_matrix
    >>> y_true =    np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    >>> y_pred = ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    ...                      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> cm = predict_accept_confusion_matrix(y_true=y_true, y_pred=y_pred)
    >>> cm
    array([[1, 2],
           [3, 4]])
    """
    y_correct = y_true == y_pred
    y_accepted = y_pred.get_dense_neg_fallback_mask()
    return confusion_matrix(
        y_correct,
        y_accepted,
        labels=labels,
        sample_weight=sample_weight,
        normalize=normalize,
    )


def _check_targets(y_true, y_pred):
    """Check that ``y_true`` and ``y_pred`` belong to the same classification task.

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predictions.

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``.
    y_true : array or indicator matrix
    y_pred : array or indicator matrix
    """
    check_consistent_length(y_true, y_pred)

    type_true = type_of_target(y_true)
    type_comb = type_of_target(y_pred)
    y_type = {type_true, type_comb}
    if y_type == {"binary", "multiclass"}:
        y_type = {"multiclass"}
    if len(y_type) > 1:
        raise ValueError(
            f"Classification metrics can't handle a mix of "
            f"{type_true} and {type_comb} targets"
        )
    y_type = y_type.pop()
    if y_type not in {"binary", "multiclass"}:
        raise ValueError(f"{y_type} is not supported")

    if y_type in {"binary", "multiclass"}:
        y_true = column_or_1d(y_true)

        if y_type == "binary":
            try:
                unique_values = np.union1d(y_true, y_pred)
            except TypeError as e:
                # We expect y_true and y_pred to be of the same data type.
                # If `y_true` was provided to the classifier as strings,
                # `y_pred` given by the classifier will also be encoded with
                # strings. So we raise a meaningful error
                raise TypeError(
                    "Labels in y_true and y_pred should be of the same type. "
                    f"Got y_true={np.unique(y_true)} and "
                    f"y_pred={np.unique(y_pred)}. Make sure that the "
                    "predictions provided by the classifier coincides with "
                    "the true labels."
                ) from e
            if unique_values.shape[0] > 2:
                y_type = "multiclass"

    return y_type, y_true, y_pred


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": [ska.FBNDArray],
    },
    prefer_skip_nested_validation=True,
)
def predict_reject_accuracy_score(y_true, y_pred):
    """Calculates the ratio of true acceptance and rejection to all predictions.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : FBNDarray
        Base estimator predictions w/ fallback mask.

    Returns
    -------
    score : (TA + TR) / (TA + TR + FA + FR)
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type.startswith("multilabel"):
        raise ValueError("Multilabel outputs are not supported.")

    reject_mask = y_pred.get_dense_fallback_mask()
    accept_mask = ~reject_mask

    true_accept = sum(y_true[accept_mask] == y_pred[accept_mask])
    true_reject = sum(y_true[reject_mask] != y_pred[reject_mask])
    false_accept = sum(y_true[accept_mask] != y_pred[accept_mask])
    false_reject = sum(y_true[reject_mask] == y_pred[reject_mask])

    try:
        return (true_accept + true_reject) / (
            true_accept + true_reject + false_accept + false_reject
        )
    except ZeroDivisionError:
        warnings.warn(
            "invalid value encountered in scalar divide",
            category=RuntimeError,
        )
        return np.nan
