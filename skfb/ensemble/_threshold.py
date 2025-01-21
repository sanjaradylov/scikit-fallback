"""Threshold-based cascade ensembles."""

from typing import Sequence

import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_X_y, NotFittedError
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


class CascadeNotFittedWarning(UserWarning):
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
        self.set_thresholds(thresholds)
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
                self._current_estimators = self.estimators_[:]
                self._current_thresholds = self.thresholds[:]
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
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=False, dtype=None)

        self.classes_ = unique_labels(y)

        if not hasattr(self, "estimators_"):
            self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(fit_one)(estimator, X, y, sample_weight)
                for estimator in self.estimators
            )
        self._current_estimators = self.estimators_[:]
        self._current_thresholds = self.thresholds[:]

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
        X = check_array(X, accept_sparse=True, ensure_2d=False, dtype=None)

        is_binary = len(self.classes_) == 2

        # Process decision scores for binary classification
        if is_binary and self.response_method == "decision_function":
            y_score = self._predict_binary_class_scores(X)
            y_pred = np.take(self.classes_, y_score >= 0.0)
            return (
                earray(y_pred, y_score.ensemble_mask) if self.return_earray else y_pred
            )
        # Process decision scores/probas for multiclass classification
        else:
            y_score = self._predict_multi_class_scores(X)
            y_pred = np.take(self.classes_, y_score.argmax(axis=1))
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
        X = check_array(X, accept_sparse=True, ensure_2d=False, dtype=None)
        return self._predict_multi_class_scores(X)

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

    def decision_function(self, X):
        """Predicts decision scores using one or more base estimators.

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
        y_score : ndarray of shape n_samples
            Decision scores predicted by the base estimators.
        """
        check_is_fitted(self, attributes="is_fitted_")
        X = check_array(X, accept_sparse=True, ensure_2d=False, dtype=None)

        is_binary = len(self.classes_) == 2
        if is_binary:
            return self._predict_binary_class_scores(X)
        else:
            return self._predict_multi_class_scores(X)

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

    def set_thresholds(self, thresholds):
        """Transforms ``thresholds`` into correct sequence of thresholds."""
        if isinstance(thresholds, float):
            self.thresholds = [thresholds] * (len(self.estimators) - 1)
        else:
            assert (
                len(thresholds) >= len(self.estimators) - 1
            ), "thresholds must be provided for at least all but the last estimator"

            self.thresholds = list(thresholds)

        if self.response_method == "decision_function":
            if len(thresholds) == len(self.estimators) - 1:
                self.thresholds.append(-np.inf)
        else:
            self.thresholds.append(0.0)

        self._current_thresholds = self.thresholds[:]

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
            self._current_thresholds = self.thresholds[:]

        elif isinstance(index, Sequence):
            self._current_estimators = [self.estimators_[i] for i in index]
            self._current_thresholds = [self.thresholds[i] for i in index[:-1]]
            if self.response_method == "predict_proba":
                self._current_thresholds.append(0.0)
            else:
                self._current_thresholds.append(-np.inf)

        elif isinstance(index, slice):
            self._current_estimators = self.estimators_[index]
            self._current_thresholds = self.thresholds[index]
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

    def _predict_multi_class_scores(self, X):
        """Estimates scores for predict_proba and multiclass decision_function."""
        # region Prevent masking overhead if only one base estimator is activated.
        if len(self._current_estimators) == 1:
            y_score = getattr(self._current_estimators[0], self.response_method)(X)
            return earray(y_score) if self.return_earray else y_score
        # endregion

        # region Store ensemble mask
        if self.return_earray:
            ensemble_mask = np.zeros(
                shape=(len(X), len(self.estimators_)),
                dtype=np.bool_,
            )
        # endregion

        # region Cascading w/ more than one base estimator
        # Composite proba/decision scores
        y_score = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        # Current mask: True means that sample should be deferred
        deferred = np.ones(len(X), dtype=np.bool_)
        if self.return_earray:
            previous_mask = np.ones_like(deferred)

        for i, (estimator, threshold) in enumerate(
            zip(self._current_estimators, self._current_thresholds)
        ):
            if X.ndim == 1:
                y_score[deferred] = getattr(estimator, self.response_method)(
                    X[deferred]
                )
            else:
                y_score[deferred] = getattr(estimator, self.response_method)(
                    X[deferred, :]
                )
            deferred[deferred] = y_score[deferred].max(axis=1) < threshold

            if self.return_earray:
                mask = ~deferred
                mask[~previous_mask] = False
                ensemble_mask[:, i] = mask
                previous_mask = deferred.copy()

            if deferred.sum() == 0:
                break

        if self.return_earray:
            return earray(y_score, ensemble_mask)
        else:
            return y_score
        # endregion

    def _predict_binary_class_scores(self, X):
        """Estimates scores for predict_proba and multiclass decision_function."""
        # region Prevent masking overhead if only one base estimator is activated.
        if len(self._current_estimators) == 1:
            y_score = getattr(self._current_estimators[0], self.response_method)(X)
            return earray(y_score) if self.return_earray else y_score
        # endregion

        # region Store ensemble mask
        if self.return_earray:
            ensemble_mask = np.zeros(
                shape=(len(X), len(self.estimators_)),
                dtype=np.bool_,
            )
        # endregion

        # region Cascading w/ more than one base estimator
        # Composite proba/decision scores
        y_score = np.zeros(len(X), dtype=np.float64)
        # Current mask: True means that sample should be deferred
        deferred = np.ones(len(X), dtype=np.bool_)
        if self.return_earray:
            previous_mask = np.ones_like(deferred)

        for i, (estimator, threshold) in enumerate(
            zip(self._current_estimators, self._current_thresholds)
        ):
            if X.ndim == 1:
                y_score[deferred] = getattr(estimator, self.response_method)(
                    X[deferred]
                )
            else:
                y_score[deferred] = getattr(estimator, self.response_method)(
                    X[deferred, :]
                )
            deferred[deferred] = y_score[deferred] < threshold

            if self.return_earray:
                mask = ~deferred
                mask[~previous_mask] = False
                ensemble_mask[:, i] = mask
                previous_mask = deferred.copy()

            if deferred.sum() == 0:
                break

        if self.return_earray:
            return earray(y_score, ensemble_mask)
        else:
            return y_score
        # endregion
