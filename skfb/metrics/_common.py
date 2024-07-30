"""Metrics for both classification and regression w/ a reject option."""

import warnings

import numpy as np

from ..core import array as ska
from ..utils._legacy import validate_params


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like", ska.FBNDArray],
        "score_func": [callable],
        "raise_warning": [bool],
    },
    prefer_skip_nested_validation=True,
)
def prediction_quality(
    y_true,
    y_pred,
    score_func,
    fallback_label=None,
    raise_warning=True,
    **kwargs,
):
    """Runs ``score_func`` on accepted samples.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_pred : array-like or FBNDArray, shape (n_samples,) or (n_samples, n_classes)
        Either array of combined predictions or estimator predictions w/ fallback mask.
        If combined, then ``fallback_label`` should be provided.
    score_func : callable
        Scoring function to call on accepted samples.
    fallback_label : any, default=None
        If predictions are combined, indicates the label of fallback.
    raise_warning : bool, default=True
        If all samples were rejected, raises a warning.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> from skfb.metrics import prediction_quality_score
    >>> y_true = np.array([1, 0, 0, 1, 0, 1])
    >>> y_pred = np.array([0, 0, -1, -1, 0, 1])
    >>> prediction_quality_score(y_true, y_pred, accuracy_score)
    0.75
    >>> y_pred = np.array([-1, -1, -1, -1, -1, -1])
    >>> prediction_quality_score(y_true, y_pred, accuracy_score,
    ...                          raise_warning=False)
    nan
    """
    if not isinstance(y_pred, ska.FBNDArray):
        non_rejected_mask = y_pred != fallback_label
    else:
        non_rejected_mask = y_pred.get_dense_neg_fallback_mask()

    if y_pred[non_rejected_mask].size <= 0:
        if raise_warning:
            warnings.warn("All examples were rejected; returning nan", UserWarning)
        return np.nan

    return score_func(y_true[non_rejected_mask], y_pred[non_rejected_mask], **kwargs)
