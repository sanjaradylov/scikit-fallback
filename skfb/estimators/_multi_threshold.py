"""Classification w/ fallback option based on class-specific thresholds."""

__all__ = (
    "multi_threshold_predict_or_fallback",
    "MultiThresholdFallbackClassifier",
)

import numpy as np

from sklearn.utils.validation import check_is_fitted, NotFittedError

# pylint: disable=import-error,no-name-in-module
# pyright: reportMissingModuleSource=false
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    Real,
    StrOptions,
    validate_params,
)

from .base import BaseFallbackClassifier
from ..core import array as ska


def _is_top_low(y_score, thresholds):
    """Whether the top confidence score is lower than the threshold."""
    hard_preds = y_score.argmax(axis=1).reshape(-1, 1)
    soft_preds = np.take_along_axis(y_score, hard_preds, 1).ravel()
    pick_thresholds = [thresholds[p[0]] for p in hard_preds]
    fallback_mask = soft_preds < pick_thresholds

    return fallback_mask


def _top_2(scores):
    """Returns top 2 scores."""
    return np.sort(scores)[-2:]


def _are_top_2_close(y_score, ambiguity_threshold):
    """Whether the scores of top two classes are lower than the ambiguity threshold."""
    y_top_2 = np.apply_along_axis(_top_2, 1, y_score)
    y_diff = np.apply_over_axes(np.diff, y_top_2, 1)
    return y_diff[:, 0] < ambiguity_threshold


@validate_params(
    {
        "estimator": [HasMethods(["fit", "predict_proba"])],
        "X": ["array-like", "sparse matrix"],
        "thresholds": ["array-like"],
        "classes": [None, np.ndarray],
        "ambiguity_threshold": [Interval(Real, left=0.0, right=1.0, closed="both")],
        "fallback_mode": [StrOptions({"return", "store"})],
        "return_probas": [bool],
    },
    prefer_skip_nested_validation=True,
)
def multi_threshold_predict_or_fallback(
    estimator,
    X,
    thresholds,
    classes=None,
    ambiguity_threshold=0.0,
    fallback_label=-1,
    fallback_mode="return",
    return_probas=False,
):
    """For every sample, either predicts a class or rejects a prediction.

    Parameters
    ----------
    estimator : object
        Fitted base estimator supporting probabilistic predictions.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input samples.
    thresholds : array-like, shape (n_classes,)
        Certainty thresholds for each class.
    classes : ndarray, shape = (n_classes,), default=None
        NDArray of class labels. Defaults to ``estimator`` classes.
    ambiguity_threshold : float, default=0.0
        Predictions w/ the close top 2 scores are rejected.
    fallback_label : any, default=-1
        Rejected samples are labeled w/ this label.
    fallback_mode : {"store", "return"}, default="return"
        If "store", returns an FBNDArray of shape (n_samples,) w/ estimator predictions
        and fallback mask attribute.
        If "return", returns an NDArray of shape (n_samples,) w/ estimator predictions
        and rejections.

    Returns
    -------
    FBNDArray or ndarray, shape = n_samples
        Either combined predictions or predictions w/ fallback mask.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skfb.estimators import multi_threshold_predict_or_fallback
    >>> X = np.array(
    ...     [
    ...         [-3, -3], [-3, -2], [-2, -3],
    ...         [2, 3], [2, 4], [3, 4], [3.1, 3], [3, 2.9],
    ...         [4, 3], [3, 2], [4, 2], [2.9, 3], [3, 3.1],
    ...     ])
    >>> y = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 5)
    >>> estimator = LogisticRegression(C=10_000, random_state=0).fit(X, y)
    >>> thresholds = {"a": 0.99, "b": 0.8, "c": 0.75}
    >>> multi_threshold_predict_or_fallback(
    ...     estimator, X, estimator.classes_, thresholds, fallback_label="d")
    array(['a', 'a', 'a', 'b', 'b', 'b', 'd', 'd', 'c', 'c', 'c', 'd', 'd'],
          dtype='<U1')
    """
    if classes is None:
        classes = estimator.classes_

    y_prob = estimator.predict_proba(X)
    fallback_mask = _is_top_low(y_prob, thresholds)
    if ambiguity_threshold > 0.0:
        fallback_mask |= _are_top_2_close(y_prob, ambiguity_threshold)

    if fallback_mode == "return":
        acceptance_mask = ~fallback_mask
        y_comb = np.empty(len(fallback_mask), dtype=classes.dtype)
        y_comb[acceptance_mask] = classes.take(y_prob[acceptance_mask].argmax(axis=1))
        y_comb[fallback_mask] = fallback_label
        return y_comb if not return_probas else (y_comb, y_prob)
    else:
        y_pred = classes.take(y_prob.argmax(axis=1))
        y_pred = ska.fbarray(y_pred, fallback_mask)
        return y_pred if not return_probas else (y_pred, y_prob)


class MultiThresholdFallbackClassifier(BaseFallbackClassifier):
    """A fallback classifier based on local thresholds.

    Parameters
    ----------
    estimator : object
        The best estimator making decisions w/o fallbacks.
    thresholds : dict-like
        Mapping from class labels to local fallback thresholds.
    ambiguity_threshold : float, default=0.0
        Predictions w/ the close top 2 scores are rejected.
    fallback_label : any, default=-1
        The label of a rejected example.
        Should be compatible w/ the class labels from training data.
    fallback_mode : {"return", "store"}, default="store"
        While predicting, whether to return a numpy ndarray of both predictions and
        fallbacks, or an fbndarray of predictions storing also fallback mask.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skfb.estimators import MultiThresholdFallbackClassifier
    >>> X = np.array(
    ...     [
    ...         [-3, -3], [-3, -2], [-2, -3],
    ...         [2, 3], [2, 4], [3, 4], [3.1, 3], [3, 2.9],
    ...         [4, 3], [3, 2], [4, 2], [2.9, 3], [3, 3.1],
    ...     ])
    >>> y = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 5)
    >>> estimator = LogisticRegression(C=10_000, random_state=0).fit(X, y)
    >>> thresholds = [0.99, 0.8, 0.75]
    >>> r = MultiThresholdFallbackClassifier(estimator, thresholds=thresholds)
    >>> r.set_params(fallback_label="d", fallback_mode="return").fit(X, y).predict(X)
    array(['a', 'a', 'a', 'b', 'b', 'b', 'd', 'd', 'c', 'c', 'c', 'd', 'd'],
          dtype='<U1')
    """

    _parameter_constraints = {**BaseFallbackClassifier._parameter_constraints}
    _parameter_constraints.update(
        {
            "thresholds": ["array-like"],
            "ambiguity_threshold": [Interval(Real, 0.0, 1.0, closed="both")],
        },
    )

    def __init__(
        self,
        estimator,
        *,
        thresholds,
        ambiguity_threshold=0.0,
        fallback_label=-1,
        fallback_mode="store",
    ):
        super().__init__(
            estimator=estimator,
            fallback_label=fallback_label,
            fallback_mode=fallback_mode,
        )
        self.thresholds = thresholds
        self.ambiguity_threshold = ambiguity_threshold
        # NOTE: I believe we are violating a scikit-learn proposal by assigning an
        #       attribute right after the initialization instead of the fitting.
        #       But this seems to be the best way to prevent refitting.
        try:
            check_is_fitted(self.estimator, "classes_")
            fallback_label_ = self._validate_fallback_label(self.estimator.classes_)
            fitted_params = {
                "estimator_": self.estimator,
                "classes_": self.estimator.classes_,
                "fallback_label_": fallback_label_,
            }
            self._set_fitted_attributes(fitted_params)
        except NotFittedError:
            pass

    def _predict(self, X):
        """Predicts classes based on the fixed class-wise certainty thresholds."""
        return multi_threshold_predict_or_fallback(
            self.estimator_,
            X,
            self.thresholds,
            classes=self.classes_,
            ambiguity_threshold=self.ambiguity_threshold,
            fallback_label=self.fallback_label_,
            fallback_mode=self.fallback_mode,
        )

    def _set_fallback_mask(self, y_prob):
        """Sets the fallback mask for predicted probabilities."""
        y_prob.fallback_mask = _is_top_low(y_prob, self.thresholds) | _are_top_2_close(
            y_prob, self.ambiguity_threshold
        )
