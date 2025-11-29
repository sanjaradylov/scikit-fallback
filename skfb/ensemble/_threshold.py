"""Threshold-based cascade ensembles."""

from typing import Sequence

import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import (
    check_array,
    check_X_y,
    NotFittedError,
)
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
    Real,
    StrOptions,
    validate_params,
)
from ._common import fit_one
from ..core.array import earray
from ..core.exceptions import SKFBWarning


class CascadeNotFittedWarning(SKFBWarning):
    """Raised if base estimators in cascade are not fitted or fitted incorrectly."""


class ThresholdCascadeClassifier(BaseEstimator, ClassifierMixin):
    """Cascade of classifiers w/ deferrals based on predefined thresholds.

    During inference, runs the first estimator and if a predicted score is lower than
    ``thresholds[0]``, tries the second, and so on. The last estimator always makes
    predictions on the samples deferred by the previous estimators.
    If every estimator is fitted, it is not necessary to run ``fit`` to make
    predictions.

    Parameters
    ----------
    estimators : array-like of object, length n_estimators
        Base estimators. Preferrably, from weakest (e.g., rule-based or linear) to
        strongest (e.g., gradient boosting).
    thresholds : float or array-like of float, length n_estimators - 1
        Deferral thresholds for each base estimator except the last.
        If only one number is specified, every estimator (except the last) will have
        the same threshold (i.e., the threshold will be *global*).
    response_method : {"predict_proba", "decision_function"}, default="predict_proba"
        Methods by ``estimators`` for which we want to find return deferral thresholds.
        For ``"decision_function"``, ``thresholds`` can be negative.
    return_earray : bool, default=False
        Whether to return :class:`~skfb.core.ENDArray` of predicted classes / scores
        or plain numpy ndarray.
    prefit : bool, default=False
        Whether estimators are fitted. If True, checks their ``classes_`` attributes
        for intercompatibility.
    n_jobs : int, default=None
        Number of parallel jobs used during training.
    verbose : int, default=False
        Verbosity level.

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.ensemble import ThresholdCascadeClassifier
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
        "thresholds": ["array-like", Interval(Real, None, None, closed="neither")],
        "response_method": [StrOptions({"decision_function", "predict_proba"})],
        "return_earray": ["boolean"],
        "prefit": ["boolean"],
        "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        estimators,
        thresholds,
        response_method="predict_proba",
        return_earray=True,
        prefit=False,
        n_jobs=None,
        verbose=False,
    ):
        self.estimators = estimators
        self.response_method = response_method
        self.thresholds = thresholds
        self._set_thresholds(thresholds)
        self.return_earray = return_earray
        self.prefit = prefit
        self.n_jobs = n_jobs
        self.verbose = verbose

        # region Check if base estimators are fitted correctly
        if self.prefit:
            classes = None
            for i, estimator in enumerate(self.estimators):
                try:
                    check_is_fitted(estimator, "classes_")

                    if not (
                        classes is None or np.array_equal(classes, estimator.classes_)
                    ):
                        warnings.warn(
                            f"Estimators {i} and {i-1} predict different classes; "
                            f"please, run cascade's `fit` method to train all "
                            f"estimators.",
                            category=CascadeNotFittedWarning,
                        )
                        break

                    classes = estimator.classes_

                except NotFittedError:
                    warnings.warn(
                        f"Estimator {i} is not fitted; "
                        f"please, run cascade's `fit` method to train all estimators.",
                        category=CascadeNotFittedWarning,
                    )
                    break

            else:
                self.estimators_ = estimators
                self.classes_ = classes
                self._current_estimators = self.estimators_[:]
                self._current_thresholds = self.thresholds_[:]
                self.is_fitted_ = True
        # endregion

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
        X, y = check_X_y(
            X,
            y,
            accept_sparse=True,
            allow_nd=True,
            dtype=None,
            ensure_2d=False,
        )

        self.classes_ = unique_labels(y)

        if not hasattr(self, "estimators_"):
            self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(fit_one)(estimator, X, y, sample_weight)
                for estimator in self.estimators
            )
        self._current_estimators = self.estimators_[:]
        self._current_thresholds = self.thresholds_[:]

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
        predictions if all the previous estimators deferred.

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
        X = check_array(
            X,
            accept_sparse=True,
            allow_nd=True,
            dtype=None,
            ensure_2d=False,
        )

        y_score = self._predict_scores(X)

        if y_score.ndim == 2:
            y_pred = np.take(self.classes_, y_score.argmax(axis=1))
        else:
            y_pred = np.take(self.classes_, y_score >= 0)

        return (
            earray(y_pred, y_score.ensemble_mask) if self.return_earray else y_pred
        )

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
        predictions if all the previous estimators deferred.

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
        X = check_array(
            X,
            accept_sparse=True,
            allow_nd=True,
            dtype=None,
            ensure_2d=False,
        )
        return self._predict_scores(X)

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
        predictions if all the previous estimators deferred.

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

    def decision_function(self, X):
        """Predicts decision scores using one or more base estimators.

        Tries estimators in the order specified during initialization. If the first
        estimator doesn't have a score higher or equal than the first threshold,
        switches to the second estimator, and so on. The last estimator always makes
        predictions if all the previous estimators deferred.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
            Must fulfill the input assumptions of the underlying estimators.

        Returns
        -------
        y_score : ndarray of shape n_samples
            Decision scores predicted by the base estimators.
        """
        check_is_fitted(self, attributes="is_fitted_")
        X = check_array(
            X,
            accept_sparse=True,
            allow_nd=True,
            dtype=None,
            ensure_2d=False,
        )
        return self._predict_scores(X)

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

    def set_params(self, **params):
        """Sets the parameters of the cascade.

        If thresholds are provided, the transformations are done accordingly, so there
        is no need to refit the cascade.

        Parameters
        ----------
        **params : dict
            Cascade parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        if "thresholds" in params:
            self._set_thresholds(params["thresholds"])
        return super().set_params(**params)

    def _set_thresholds(self, thresholds):
        """Transforms ``thresholds`` into correct sequence of thresholds."""
        if isinstance(thresholds, float):
            self.thresholds_ = [thresholds] * (len(self.estimators) - 1)
        else:
            assert (
                len(thresholds) >= len(self.estimators) - 1
            ), "thresholds must be provided for at least all but the last estimator"

            self.thresholds_ = list(thresholds)

        if self.response_method == "decision_function":
            if len(thresholds) == len(self.estimators) - 1:
                self.thresholds_.append(-np.inf)
        else:
            self.thresholds_.append(0.0)

        self._current_thresholds = self.thresholds_[:]

        return self

    def set_estimators(self, index):
        """Sets the estimators and thresholds to use for prediction and scoring.

        If a single index passed, the corresponding threshold is set to 0.0 or -np.inf
        depending on the ``response_method`` attribute.

        By default, uses all trained estimators (available by the ``estimators_``
        ``thresholds`` attribute).

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
            If ``index`` is of unsupported type or value.
        """
        check_is_fitted(self, attributes="is_fitted_")

        if isinstance(index, int):
            self._current_estimators = [self.estimators_[index]]
            if self.response_method == "predict_proba":
                self._current_thresholds = [0.0]
            else:
                self._current_thresholds = [-np.inf]

        elif index == "all":
            self._current_estimators = self.estimators_[:]
            self._current_thresholds = self.thresholds_[:]

        elif isinstance(index, Sequence):
            self._current_estimators = [self.estimators_[i] for i in index]
            self._current_thresholds = [self.thresholds_[i] for i in index[:-1]]
            if self.response_method == "predict_proba":
                self._current_thresholds.append(0.0)
            else:
                self._current_thresholds.append(-np.inf)

        elif isinstance(index, slice):
            self._current_estimators = self.estimators_[index]
            self._current_thresholds = self.thresholds_[index]
            self._current_thresholds.pop()
            if self.response_method == "predict_proba":
                self._current_thresholds.append(0.0)
            else:
                self._current_thresholds.append(-np.inf)

        else:
            raise TypeError(
                f"index must be int or slice or sequence of int, not {type(index)}"
            )

        return self

    def reset_estimators(self):
        """Reactivates all the base estimators.

        Same as ``set_estimators("all")``. Use if you previously set to skip some
        estimators and thresholds, and want to activate all estimators again.

        Returns
        -------
        self : object
            Returns self.
        """
        return self.set_estimators("all")

    def _predict_scores(self, X):
        """Estimates confidence scores for `predict_proba`.

        Returns
        -------
        y_score : np.ndarray, shape = (n_samples, n_classes) or n_samples
            Confidence scores (probabilities or decision scores, depending on
            `self.response_method`).
        """
        n_samples = X.shape[0]
        n_estimators = len(self._current_estimators)
        n_classes = self._current_estimators[0].classes_.shape[0]

        # Scores to return
        if self.response_method == "predict_proba":
            y_score = np.zeros((n_samples, n_classes), dtype=np.float64)
        else:
            y_score = np.zeros(n_samples, dtype=np.float64)
        # Ensemble mask if `self.return_earray` is True
        if self.return_earray:
            ensemble_mask = np.zeros((n_samples, n_estimators), dtype=np.bool_)
        # The current sample indices to process
        remaining_idx = np.arange(n_samples)

        # region Cascaded prediction of the current selected estimators
        for i, (estimator, threshold) in enumerate(
            zip(self._current_estimators, self._current_thresholds)
        ):
            # All samples are processed
            if len(remaining_idx) == 0:
                break

            # region Predict currently deferred samples
            X_remaining = X[remaining_idx]
            y_score_remaining = getattr(estimator, self.response_method)(X_remaining)
            if self.response_method == "predict_proba":
                max_score = np.max(y_score_remaining, axis=1)
            else:
                max_score = y_score_remaining
            # endregion

            # region Mask selected samples
            if i < n_estimators - 1:
                confident_mask = max_score >= threshold
            else:
                confident_mask = np.ones(len(remaining_idx), dtype=np.bool_)
            # endregion

            # region Update indices of deferred samples
            confident_idx = remaining_idx[confident_mask]
            y_score[confident_idx] = y_score_remaining[confident_mask]
            ensemble_mask[confident_idx, i] = True
            remaining_idx = remaining_idx[~confident_mask]
            # endregion
        # endregion

        if self.return_earray:
            return earray(y_score, ensemble_mask)
        else:
            return y_score
