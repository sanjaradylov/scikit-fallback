"""Metrics for both classification and regression w/ a reject option."""

import warnings

import numpy as np
from sklearn.metrics import auc, get_scorer
from sklearn.utils import check_consistent_length

from ..core import array as ska
from ..core.exceptions import SKFBWarning
from ..utils._legacy import Interval, Real, validate_params


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
            warnings.warn("All examples were rejected; returning nan", SKFBWarning)
        return np.nan

    return score_func(y_true[non_rejected_mask], y_pred[non_rejected_mask], **kwargs)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "y_score": ["array-like", None],
        "scoring": [str, callable],
        "n_bins": [int, None],
        "min_coverage": [Interval(Real, 0.0, 1.0, closed="both")],
        "max_coverage": [Interval(Real, 0.0, 1.0, closed="both")],
        "labels": ["array-like", None],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def utility_coverage_curve(
    y_true,
    y_pred,
    *,
    y_score=None,
    scoring="accuracy",
    n_bins=None,
    min_coverage=0.05,
    max_coverage=1.0,
    labels=None,
    sample_weight=None,
):
    """Returns performance-coverage sequence:

    .. math::
        \\text{utility}(k) = \\text{scoring}(y_{true}[:k], y_{pred}[:k]),
        \\text{coverage}(k) = \\frac{k}{n_{samples}}

    where :math:`k` is the number of accepted samples; ``y_true`` and ``y_pred`` are
    sorted by confidence score in descending order.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_pred : array-like, shape (n_samples,) or (n_samples, n_classes)
        Predicted labels or probability matrix. If 2D and ``y_score`` is None,
        confidence scores are inferred as ``np.max(y_pred, axis=1)`` and hard
        predictions as ``labels[np.argmax(y_pred, axis=1)]`` (or ``np.argmax``
        if ``labels`` is None).
    y_score : array-like, shape (n_samples,), default=None
        Confidence scores for each prediction. If None, inferred from
        ``y_pred`` (requires ``y_pred`` to be 2D).
    scoring : callable or str, default="accuracy"
        Scorer as risk evaluation (e.g., log-loss, accuracy).
        Can be a scikit-learn scorer name (e.g., "accuracy", "f1") or a callable
        with signature ``scorer(y_true, y_pred) -> float`` (higher is better).
    n_bins : int, default=None
        Number of evenly spaced coverage levels to evaluate at. If None,
        evaluation happens at each unique score threshold. For example,
        ``n_bins=10`` evaluates at coverage = 0.1, 0.2, ..., 1.0.
    min_coverage : float, default=0.05
        Smallest coverage level to report. Utility estimated on very few accepted
        samples has high variance, so the low-coverage tail is dropped by default.
    max_coverage : float, default=1.0
        Largest coverage level to report.
    labels : array-like, shape (n_classes,), default=None
        Class labels ordered by column index in ``y_pred`` when ``y_pred`` is 2D.
        Used to map argmax indices to actual label values.
    sample_weight : array-like, shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    utility : ndarray, shape (n_thresholds,)
        Utility values for each obtained coverage level.
    coverage : ndarray, shape (n_thresholds,)
        Coverage values computed at each unique threshold.
    thresholds : ndarray, shape (n_thresholds,)
        Unique score thresholds in descending order.

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.metrics import utility_coverage_curve
    >>> y_true = np.array([0, 1, 0, 2, 2, 1, 0, 0, 1, 0])
    >>> y_proba = np.array([
    ...     [0.95, 0.03, 0.02],
    ...     [0.40, 0.35, 0.25],
    ...     [0.90, 0.05, 0.05],
    ...     [0.05, 0.85, 0.10],
    ...     [0.10, 0.10, 0.80],
    ...     [0.20, 0.60, 0.20],
    ...     [0.20, 0.20, 0.60],
    ...     [0.70, 0.20, 0.10],
    ...     [0.70, 0.20, 0.10],
    ...     [0.55, 0.25, 0.20],
    ... ])
    >>> utility, coverage, thresholds = utility_coverage_curve(y_true, y_proba)
    >>> utility
    array([1.0, 1.0, 0.67, 0.75, 0.67, 0.625, 0.67, 0.6])
    >>> coverage
    array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0])
    >>> thresholds
    array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.4])
    """
    if min_coverage >= max_coverage:
        raise ValueError("min_coverage should be less than max_coverage")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine if the scorer expects probabilities rather than hard labels.
    _needs_proba = False
    if isinstance(scoring, str):
        _scorer_tmp = get_scorer(scoring)
        _resp = getattr(_scorer_tmp, "_response_method", "predict")
        if isinstance(_resp, str):
            _needs_proba = _resp == "predict_proba"
        else:
            _needs_proba = "predict_proba" in _resp

    if y_score is None:
        if y_pred.ndim < 2:
            raise ValueError(
                "y_score must be provided when y_pred is not a 2D probability matrix."
            )
        y_score = np.max(y_pred, axis=1)
        if not _needs_proba:
            if labels is not None:
                labels = np.asarray(labels)
                y_pred = labels[np.argmax(y_pred, axis=1)]
            else:
                y_pred = np.argmax(y_pred, axis=1)
    else:
        y_score = np.asarray(y_score)

    check_consistent_length(y_true, y_pred, y_score)

    # Infer all unique class labels from full y_true for scorers that need them
    # (e.g., log_loss requires knowing all classes even on subsets).
    all_labels = np.unique(y_true)

    if isinstance(scoring, str):
        scorer = get_scorer(scoring)

        def _score_fn(y_true_prefix, y_pred_prefix, sample_weight_prefix):
            kwargs = dict(scorer._kwargs)
            if sample_weight_prefix is not None:
                kwargs["sample_weight"] = sample_weight_prefix
            kwargs.setdefault("labels", all_labels)
            try:
                return scorer._sign * scorer._score_func(
                    y_true_prefix,
                    y_pred_prefix,
                    **kwargs,
                )
            except TypeError:
                # Score function doesn't accept `labels`; retry without it.
                kwargs.pop("labels", None)
                return scorer._sign * scorer._score_func(
                    y_true_prefix,
                    y_pred_prefix,
                    **kwargs,
                )

    else:

        def _score_fn(y_true_prefix, y_pred_prefix, sample_weight_prefix):
            if sample_weight_prefix is None:
                try:
                    return scoring(y_true_prefix, y_pred_prefix, labels=all_labels)
                except TypeError:
                    return scoring(y_true_prefix, y_pred_prefix)
            try:
                return scoring(
                    y_true_prefix,
                    y_pred_prefix,
                    sample_weight=sample_weight_prefix,
                    labels=all_labels,
                )
            except TypeError:
                try:
                    return scoring(
                        y_true_prefix,
                        y_pred_prefix,
                        sample_weight=sample_weight_prefix,
                    )
                except TypeError:
                    return scoring(y_true_prefix, y_pred_prefix)

    sorted_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[sorted_indices]
    y_pred = y_pred[sorted_indices]
    y_score = y_score[sorted_indices]

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)[sorted_indices]
    else:
        sample_weight = None

    # Evaluate once per unique threshold so tied confidence scores map to one point.
    threshold_end_indices = np.r_[
        np.where(np.diff(y_score) != 0)[0],
        len(y_score) - 1,
    ]

    if n_bins is not None:
        # Evaluate at evenly spaced coverage levels.
        n_samples = len(y_score)
        target_coverage = np.linspace(1.0 / n_bins, 1.0, n_bins)
        target_indices = np.clip(
            np.ceil(target_coverage * n_samples).astype(int) - 1,
            0,
            n_samples - 1,
        )
        # Snap each target index forward to the end of its tie group.
        for i, idx in enumerate(target_indices):
            # Find the tie group that contains this index.
            group_mask = threshold_end_indices >= idx
            if group_mask.any():
                target_indices[i] = threshold_end_indices[group_mask][0]
        # Deduplicate while preserving order.
        _, unique_mask = np.unique(target_indices, return_index=True)
        target_indices = target_indices[np.sort(unique_mask)]
        eval_indices = target_indices

    else:
        eval_indices = threshold_end_indices

    coverage = (eval_indices + 1) / len(y_score)
    in_range = (coverage >= min_coverage) & (coverage <= max_coverage)
    if not in_range.any():
        raise ValueError(
            f"No coverage level falls within [{min_coverage}, {max_coverage}]; "
            "widen the range or provide more samples"
        )
    eval_indices = eval_indices[in_range]
    coverage = coverage[in_range]
    thresholds = y_score[eval_indices]

    utility = np.empty_like(coverage, dtype=float)
    for i, end_idx in enumerate(eval_indices):
        upto = end_idx + 1
        if sample_weight is None:
            utility[i] = _score_fn(y_true[:upto], y_pred[:upto], None)
        else:
            utility[i] = _score_fn(y_true[:upto], y_pred[:upto], sample_weight[:upto])

    return utility, coverage, thresholds


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "y_score": ["array-like", None],
        "scoring": [str, callable],
        "min_coverage": [Interval(Real, 0.0, 1.0, closed="both")],
        "max_coverage": [Interval(Real, 0.0, 1.0, closed="both")],
        "labels": ["array-like", None],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def utility_coverage_auc_score(
    y_true,
    y_pred,
    *,
    y_score=None,
    scoring="accuracy",
    min_coverage=0.05,
    max_coverage=1.0,
    labels=None,
    sample_weight=None,
):
    """Evaluates area under the utility-coverage curve.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_pred : array-like, shape (n_samples,) or (n_samples, n_classes)
        Predicted labels or probability matrix. If 2D and ``y_score`` is None,
        confidence scores and hard predictions are inferred from this matrix.
    y_score : array-like, shape (n_samples,), default=None
        Confidence scores for each prediction. If None, inferred from
        ``y_pred`` (requires ``y_pred`` to be 2D).
    scoring : str or callable, default="accuracy"
        A string (see :ref:`scoring_parameter`) or a scorer callable object /
        function with signature ``scorer(y_true, y_pred)``.
    min_coverage : float, default=0.05
        Smallest coverage level to integrate from.
    max_coverage : float, default=1.0
        Largest coverage level to integrate to.
    labels : array-like, shape (n_classes,), default=None
        Class labels ordered by column index in ``y_pred`` when ``y_pred`` is 2D.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    auc_score : float
        Area under the utility-coverage curve over
        ``[min_coverage, max_coverage]``.
    """
    utility, coverage, _ = utility_coverage_curve(
        y_true,
        y_pred,
        y_score=y_score,
        scoring=scoring,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        labels=labels,
        sample_weight=sample_weight,
    )
    return auc(coverage, utility)
