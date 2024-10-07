"""Classification metrics w/ a rejection option."""

__all__ = (
    "predict_accept_confusion_matrix",
    "predict_reject_accuracy_score",
    "predict_reject_recall_score",
)

import functools
import warnings

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    get_scorer,
    get_scorer_names,
    make_scorer,
    zero_one_loss,
)

from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.utils.multiclass import type_of_target

import numpy as np

from ..core import array as ska
from ..utils._legacy import (
    Interval,
    Real,
    StrOptions,
    validate_params,
)
from ._common import prediction_quality


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

    y_pred : FBNDArray of shape (n_samples,)
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
    score : float
        (TA + TR) / (TA + TR + FA + FR)

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.core import array as ska
    >>> from skfb.metrics import predict_reject_accuracy_score
    >>> y_true =    np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    >>> y_pred = ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    ...                      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> predict_reject_accuracy_score(y_true, y_pred)
    np.float64(0.5)
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


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": [ska.FBNDArray],
        "beta": [Interval(Real, left=0.0, right=1.0, closed="both")],
    },
    prefer_skip_nested_validation=True,
)
def predict_reject_recall_score(y_true, y_pred, beta=0.5):
    """Calculates weighted average of prediction and rejection recalls.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : FBNDarray
        Base estimator predictions w/ fallback mask.
    beta : float, default=0.5
        The weight of prediction recall.

    Returns
    -------
    score : float
        TA / (TA + FR) * beta + TR / (TR + FA) * (1 - beta)

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.core import array as ska
    >>> from skfb.metrics import predict_reject_recall_score
    >>> y_true =    np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    >>> y_pred = ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    ...                      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> # TR = 1, FA = 2, TA = 4, FR = 3
    >>> predict_reject_recall_score(y_true, y_pred, beta=0.75)
    0.5119...
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_type.startswith("multilabel"):
        raise ValueError("Multilabel outputs are not supported.")

    cm = predict_accept_confusion_matrix(y_true, y_pred)
    ta, tr, fa, fr = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]
    return ta / (ta + fr) * beta + tr / (tr + fa) * (1 - beta)


def error_rejection_loss(
    y_true,
    y_prob,
    *,
    thresholds,
    y_pred=None,
    score_func=None,
    class_weight=None,
):
    """Computes weighted combination of rejection probabilities and prediction error.

    First, for evey class from ``y_pred``, compute the fallback rate, then take weighted
    average of fallback rates. Second, add prediction error to the obtained score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities.
        Defaults to zero-one loss.
    thresholds : array-like of shape (n_classes,)
        Local (class-wise) thresholds.
    y_pred : array-like of shape (n_samples,), default=None
        Predicted classes or rejections.
        ``score_func`` can accept either ``y_prob`` or ``y_pred``.
    score_func : callable, default=None
        Prediction error scorer.
        Defaults to ``sklearn.metrics.zero_one_loss``. If ``y_pred`` is also None,
        calculates 0-1 loss between ``y_true`` and hard predictions w/o fallbacks.
        Keep in mind that greater values mean higher overall loss.
    class_weight : dict, default=None
        Mapping from classes and fallback label to weights for weighted average of
        losses. If None, defaults to classes from 0 to len(classes) mapped into
        uniform proba.

    Returns
    -------
    float : prediction-error--fallback-rate loss

    Raises
    ------
    ValueError:
        If all ``thresholds``, ``class_weight``, and ``classes`` are None.
    """
    # region Validate and set defaults to threshold, classes, and class weights
    if class_weight is None:
        n_classes = len(thresholds)
        class_weight = np.array([1 / (n_classes + 1)] * (n_classes + 1))
    # endregion

    # region Validate scoring function
    if score_func is None:
        score_func = zero_one_loss
        if y_pred is None:
            y_pred = y_prob.argmax(axis=1)
        elif isinstance(y_pred, ska.FBNDArray):
            score_func = make_scorer(
                predict_accept_confusion_matrix, greater_is_better=False
            )
    # endregion

    pred_class_mask = y_prob == y_prob.max(axis=1).reshape(-1, 1)
    fallback_mask = np.less(y_prob, thresholds)
    mask = pred_class_mask & fallback_mask

    score = mask.sum(axis=0) / pred_class_mask.sum(axis=0)
    score = sum(score * class_weight[:-1])

    if y_pred is None:
        error = score_func(y_true, y_prob)
    else:
        error = score_func(y_true, y_pred)

    score += error * class_weight[-1]

    return score


@validate_params(
    {
        "scoring": [callable, StrOptions(set(get_scorer_names())), None],
        "fallback_mode": [StrOptions({"store", "return"})],
    },
    prefer_skip_nested_validation=True,
)
def get_scoring(scoring=None, fallback_label=-1, fallback_mode="store"):
    """Returns the default scorer if it's None; otherwise, ``scoring`` itself."""
    if scoring is None:
        if fallback_mode == "store":
            scoring = predict_reject_accuracy_score
        else:
            scoring = functools.partial(
                prediction_quality,
                score_func=accuracy_score,
                fallback_label=fallback_label,
            )
    elif isinstance(scoring, str):
        scoring = get_scorer(scoring)
    return scoring


@validate_params(
    {
        "y_true": ["array-like"],
        "y_prob": [ska.FBNDArray, "array-like"],
        "classes": [None, "array-like"],
        "as_frame": [bool],
    },
    prefer_skip_nested_validation=False,
)
def describe_confidences(y_true, y_prob, classes=None, as_frame=False):
    """Returns min, max, mean, std of correct and incorrect predictions per class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities.
    classes : array-like of shape (n_classes,), default=None
        Ordered class labels: ``classes[i]`` corresponds to ``y_prob[:, i]``.
        Defaults to the unique labels from ``y_true``.
    as_frame : bool, default=False
        Whether to return pandas DataFrame of results.

    Returns
    -------
    dict or pandas.DataFrame
    """
    if classes is None:
        classes = np.unique(y_true)

    confidences = {}

    for i, c in enumerate(classes):
        class_mask = y_true == c
        c_prob = y_prob[class_mask]
        true_mask = c_prob.argmax(axis=1) == i

        max_true_prob = c_prob[true_mask].max(axis=1)
        if max_true_prob.shape[0] == 0:
            confidences[c] = [np.nan] * 4
        else:
            confidences[c] = [
                max_true_prob.min(),
                max_true_prob.max(),
                max_true_prob.mean(),
                max_true_prob.std(),
            ]

        max_false_prob = c_prob[~true_mask].max(axis=1)
        if max_false_prob.shape[0] == 0:
            confidences[c].extend([np.nan] * 4)
        else:
            confidences[c].extend(
                [
                    max_false_prob.min(),
                    max_false_prob.max(),
                    max_false_prob.mean(),
                    max_false_prob.std(),
                ]
            )

    if as_frame:
        try:
            # pylint: disable=import-outside-toplevel
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "`as_frame=True` requires pandas. You can install pandas with "
                "`pip install pandas`"
            ) from e
        columns = [
            "min_correct",
            "max_correct",
            "mean_correct",
            "std_correct",
            "min_incorrect",
            "max_incorrect",
            "mean_incorrect",
            "std_incorrect",
        ]
        return pd.DataFrame(confidences, index=columns).T
    else:
        return confidences
