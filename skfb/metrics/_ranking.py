"""Metrics to assess performance based on scores."""

import collections
import warnings

import numpy as np

from sklearn.metrics import accuracy_score, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array

from sklearn.utils.multiclass import type_of_target

from ..core.array import fbarray
from ..core.exceptions import SKFBWarning
from ..utils._legacy import (
    Interval,
    Real,
    StrOptions,
    validate_params,
)
from ._common import prediction_quality


FallbackQualityResult = collections.namedtuple(
    "FallbackQualityResult",
    "fallback_rates, scores, thresholds",
)

_FQ_DEPRECATION_MSG = (
    "{name} is deprecated in 0.3; use skfb.metrics.{replacement} instead "
    "(coverage = 1 - fallback rate)."
)


def _fallback_quality_curve(
    y_true,
    y_score,
    score_func=accuracy_score,
    predict_method="predict",
    min_fallback_rate=0.0,
    max_fallback_rate=0.95,
    raise_warning=True,
):
    """Shared implementation of the deprecated fallback-quality helpers."""
    # region Check fallback constraints
    if min_fallback_rate >= max_fallback_rate:
        raise ValueError("min_fallback_rate should be less than max_fallback_rate")
    # endregion

    # region Check types and normalize
    y_type = type_of_target(y_true, input_name="y_true")
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)

    if y_type not in {"binary", "multiclass"}:
        raise ValueError(f"{y_type} is not supported")

    y_true = LabelEncoder().fit_transform(y_true)
    # endregion

    # region Get unique thresholds
    confidence = y_score.max(axis=1)
    thresholds = np.unique(confidence)
    thresholds = np.sort(thresholds, kind="mergesort")
    # endregion

    # region Calculate fallback rate and prediction quality for every threshold
    scores, fallback_rates = [], []
    for threshold in thresholds:
        fallback_mask = confidence < threshold

        fallback_rate = fallback_mask.sum() / len(confidence)
        if not min_fallback_rate <= fallback_rate <= max_fallback_rate:
            continue

        fallback_rates.append(fallback_rate)

        if predict_method == "predict_proba":
            y_fb = fbarray(y_score[:, 1], fallback_mask)
        elif predict_method == "predict_log_proba":
            y_fb = fbarray(np.log(y_score[:, 1]), fallback_mask)
        else:
            y_fb = fbarray(y_score.argmax(axis=1), fallback_mask)

        try:
            score = prediction_quality(y_true, y_fb, score_func)
        except ValueError as err:
            if raise_warning:
                warnings.warn(
                    f"Raised ValueError('{err.args[0]}'); skipping the threshold",
                    category=SKFBWarning,
                )
            fallback_rates.pop()
            continue
        else:
            scores.append(score)
    # endregion

    assert fallback_rates, "No fallback rate collected; reset fallback constraints"
    assert scores, "No scores calculated; check input arguments"

    return FallbackQualityResult(fallback_rates, scores, thresholds)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
        "score_func": [callable],
        "predict_method": [
            StrOptions({"predict", "predict_proba", "predict_log_proba"}),
        ],
        "min_fallback_rate": [Interval(Real, 0.0, 1.0, closed="both")],
        "max_fallback_rate": [Interval(Real, 0.0, 1.0, closed="both")],
        "raise_warning": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def fallback_quality_curve(
    y_true,
    y_score,
    score_func=accuracy_score,
    predict_method="predict",
    min_fallback_rate=0.0,
    max_fallback_rate=0.95,
    raise_warning=True,
):
    """Constructs prediction quality vs fallback rate curve.

    .. deprecated:: 0.3
        Use :func:`~skfb.metrics.utility_coverage_curve` instead, where
        ``coverage = 1 - fallback_rate``.

    First, determines unique thresholds on input samples, then for every threshold,
    calculates the fallback rate and prediction quality.

    Parameter
    ---------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_score : array-like, shape (n_samples,)
        Predicted scores.
    score_func : callable, default=sklearn.metrics.accuracy_score
        Scoring function (such as accuracy score) to calculate on accepted
        examples for every fallback threshold.
    predict_method : {"predict", "predict_proba", "predict_log_proba"}, \
            default="predict"
        Whether ``score_func`` accepts classes, probabilities, or log-probabilities.
    min_fallback_rate : float, default=0.0
        Minimum fallback rate to include.
    max_fallback_rate : float, default=0.95
        Maximum fallback rate to include.
    raise_warning : bool, default=True
        Raise warning if ``score_func`` raises ValueError.

    Returns
    -------
    FallbackQualityResult
        Fallback rates, scores, and thresholds. All are ndarrays of the same shape
        (n_unique_thresholds,).

    Notes
    -----
    If ``score_func`` accepts probabilities, we pass the probabilities of the positive
    class.
    """
    warnings.warn(
        _FQ_DEPRECATION_MSG.format(
            name="fallback_quality_curve", replacement="utility_coverage_curve"
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return _fallback_quality_curve(
        y_true,
        y_score,
        score_func=score_func,
        predict_method=predict_method,
        min_fallback_rate=min_fallback_rate,
        max_fallback_rate=max_fallback_rate,
        raise_warning=raise_warning,
    )


@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
        "score_func": [callable],
        "predict_method": [
            StrOptions({"predict", "predict_proba", "predict_log_proba"}),
        ],
        "min_fallback_rate": [Interval(Real, 0.0, 1.0, closed="both")],
        "max_fallback_rate": [Interval(Real, 0.0, 1.0, closed="both")],
        "raise_warning": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def fallback_quality_auc_score(
    y_true,
    y_score,
    score_func=accuracy_score,
    predict_method="predict",
    min_fallback_rate=0.0,
    max_fallback_rate=0.95,
    raise_warning=True,
):
    """Returns area under prediction quality-fallback rate curve.

    .. deprecated:: 0.3
        Use :func:`~skfb.metrics.utility_coverage_auc_score` instead. Area is
        invariant to the ``coverage = 1 - fallback_rate`` reflection, so the two
        scores agree.

    First, determines unique thresholds on input samples, then for every threshold,
    calculates the fallback rate and prediction quality.

    Parameter
    ---------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_score : array-like, shape (n_samples,)
        Predicted scores.
    score_func : callable, default=sklearn.metrics.accuracy_score
        Scoring function (such as accuracy score) to calculate on accepted
        examples for every fallback threshold.
    predict_method : {"predict", "predict_proba", "predict_log_proba"},
            default="predict"
        Whether ``score_func`` accepts classes, probabilities, or log-probabilities.
    min_fallback_rate : float, default=0.0
        Minimum fallback rate to include.
    max_fallback_rate : float, default=0.95
        Maximum fallback rate to include.
    raise_warning : bool, default=True
        Raise warning if ``score_func`` raises ValueError.

    Returns
    -------
    float : The area under the prediction quality vs fallback rate curve.
    """
    warnings.warn(
        _FQ_DEPRECATION_MSG.format(
            name="fallback_quality_auc_score",
            replacement="utility_coverage_auc_score",
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    fallback_rates, scores, _ = _fallback_quality_curve(
        y_true,
        y_score,
        score_func=score_func,
        predict_method=predict_method,
        min_fallback_rate=min_fallback_rate,
        max_fallback_rate=max_fallback_rate,
        raise_warning=raise_warning,
    )
    return auc(fallback_rates, scores)
