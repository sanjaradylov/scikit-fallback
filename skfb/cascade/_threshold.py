"""Threshold-based cascade ensembles."""

from typing import Sequence

import inspect

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted, clone
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels

try:
    from sklearn.utils.parallel import delayed, Parallel
except ModuleNotFoundError:
    from joblib import Parallel

    # pylint: disable=ungrouped-imports
    from sklearn.utils.fixes import delayed

from ..utils._legacy import (
    _fit_context,
    Integral,
    Interval,
    validate_params,
)


class ThresholdCascadeClassifier(BaseEstimator, ClassifierMixin):
    """Cascade of classifiers w/ deferrals based on predefined thresholds.

    Trains all estimators. During inference, runs the first estimator and if
    a predicted score is lower than ``thresholds[0]``, tries the second, and so on.
    The last estimator always makes predictions on the samples deferred by the previous
    estimators.

    Parameters
    ----------
    estimators : array-like of object, length n_estimators
        Base estimators. Preferrably, from weakest (e.g., rule-based or linear) to
        strongest (e.g., gradient boosting).
    thresholds : float or array-like of float, length n_estimators - 1
        Deferral thresholds for each base estimators except the first.
    n_jobs : int, default=None
        Number of parallel jobs used during training.
    verbose : int, default=False
        Verbosity level.

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.cascade import ThresholdCascadeClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = np.array([
    ...     [0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2., 2.5], [2., 2.], [2.5, 2.5]
    ... ])
    >>> y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    >>> maxent = LogisticRegression(random_state=0)
    >>> rf = RandomForestClassifier(random_state=0)
    >>> cascade = ThresholdCascadeClassifier([maxent, rf], [0.8]).fit(X, y)
    >>> cascade.score(X, y)
    1.0
    >>> cascade.set_estimators(0).score(X, y)  # Use only LogisticRegression
    0.75

    Notes
    -----
    If you want to have a fallback option (for the last estimator), consider rejectors
    from :mode:`skfb.estimators`.
    """

    _parameter_constraints = {
        "estimators": ["array-like"],
        "thresholds": ["array-like"],
        "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
        "verbose": ["verbose"],
    }

    def __init__(self, estimators, thresholds, n_jobs=None, verbose=False):
        self.estimators = estimators

        if isinstance(thresholds, float):
            thresholds = [thresholds] * (len(estimators) - 1)
        self.thresholds = thresholds

        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=False)
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "y": ["array-like"],
            "sample_weight": ["array-like", None],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X, y, sample_weight=None):
        """Fits base estimators and sets meta-estimator attributes.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_ = unique_labels(y)

        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_one)(estimator, X, y, sample_weight)
            for estimator in self.estimators
        )
        self._current_estimators = self.estimators_[:]

        self.is_fitted_ = True

        return self

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(self, X):
        """Predicts classes using one or more base estimators.

        Tries estimators in the order specified during initialization. If the first
        estimator doesn't have a score higher or equal than the first threshold,
        switches to the second estimator, and so on. The last estimator always makes
        predictions.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
            Must fulfill the input assumptions of the underlying estimators.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Classes predicted by the base estimators.
        """
        check_is_fitted(self, attributes="is_fitted_")

        acceptance_mask = np.ones(len(X), dtype=bool)
        thresholds = list(self.thresholds) + [0.0]
        y_prob = np.zeros((len(X), len(self.classes_)), dtype=float)

        for estimator, threshold in zip(self._current_estimators, thresholds):
            y_prob[acceptance_mask] = estimator.predict_proba(X[acceptance_mask, :])
            acceptance_mask ^= y_prob.max(axis=1) >= threshold

            if acceptance_mask.sum() == 0:
                break

        return y_prob.argmax(axis=1)

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_proba(self, X):
        """Predicts probabilities using one or more base estimators.

        Tries estimators in the order specified during initialization. If the first
        estimator doesn't have a score higher or equal than the first threshold,
        switches to the second estimator, and so on. The last estimator always makes
        predictions.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
            Must fulfill the input assumptions of the underlying estimators.

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            Probabilities predicted by the base estimators.
        """
        check_is_fitted(self, attributes="is_fitted_")

        acceptance_mask = np.ones(len(X), dtype=bool)
        thresholds = list(self.thresholds) + [0.0]
        y_prob = np.zeros((len(X), len(self.classes_)), dtype=float)

        for estimator, threshold in zip(self._current_estimators, thresholds):
            y_prob[acceptance_mask] = estimator.predict_proba(X[acceptance_mask, :])
            acceptance_mask ^= y_prob.max(axis=1) >= threshold

        return y_prob

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_log_proba(self, X):
        """Predicts log-probabilities using one or more base estimators.

        Tries estimators in the order specified during initialization. If the first
        estimator doesn't have a score higher or equal than the first threshold,
        switches to the second estimator, and so on. The last estimator always makes
        predictions.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
            Must fulfill the input assumptions of the underlying estimators.

        Returns
        -------
        y_score : ndarray of shape (n_samples, n_classes)
            Log-probabilities predicted by the base estimators.
        """
        return np.log(self.predict_proba(X))

    def decision_function(self, _):
        """TODO: Will be implemented for thresholds supporting non-probabilities.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Not available for now; consider predict_log_proba.")

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "y": ["array-like"],
            "sample_weight": ["array-like", None],
        },
        prefer_skip_nested_validation=True,
    )
    def score(self, X, y, sample_weight=None):
        """Computes accuracy score on true labels and cascade predictions.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to evaluate.
            Must fulfill the input assumptions of the underlying estimators.
        y : array-like of shape (n_samples,)
            True labels for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        score : float
            Accuracy score.
        """
        check_is_fitted(self, attributes="is_fitted_")

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def set_estimators(self, index):
        """Sets the estimators to use for prediction and scoring.

        By default, uses all trained estimators (available by the ``estimators_``
        attribute), but can be changed to a subset of the estimators.

        Parameters
        ----------
        index : int, or slice, or "all", or array-like of int

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        TypeError:
            If ``index`` is of unsupported type.
        """
        check_is_fitted(self, attributes="is_fitted_")

        if isinstance(index, int):
            self._current_estimators = [self.estimators_[index]]
        elif isinstance(index, Sequence):
            self._current_estimators = [self.estimators_[i] for i in index]
        elif isinstance(index, slice):
            self._current_estimators = self.estimators_[index]
        elif index == "all":
            self._current_estimators = self.estimators_[:]
        else:
            raise TypeError(
                f"index must be int or slice or sequence of int, not {type(index)}"
            )

        return self


def _fit_one(estimator, X, y, sample_weight=None):
    """Trains ``estimator`` on ``(X, y)``."""
    estimator = clone(estimator)
    if "sample_weight" in inspect.getfullargspec(estimator.fit).args:
        return estimator.fit(X, y, sample_weight=sample_weight)
    else:
        return estimator.fit(X, y)
