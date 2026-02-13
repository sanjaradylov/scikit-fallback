"""Threshold-based cascade ensembles."""

from typing import Sequence

import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted, clone
from sklearn.metrics import accuracy_score, get_scorer, get_scorer_names
from sklearn.model_selection import check_cv, ParameterGrid
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import NotFittedError

try:
    from sklearn.utils.parallel import delayed, Parallel
except ModuleNotFoundError:
    from joblib import Parallel, delayed

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
from ..core.exceptions import SKFBException, SKFBWarning


class CascadeNotFittedWarning(SKFBWarning):
    """Raised if base estimators in cascade are not fitted or fitted incorrectly."""


class CascadeParetoConfigWarning(SKFBWarning):
    """Raised if no Pareto configuration satisfies cost-performance constraints."""


class CascadeParetoConfigException(SKFBException):
    """Raised if no Pareto configuration satisfies cost-performance constraints."""


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
        n_samples = len(X)
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
            X_remaining = np.take(X, remaining_idx, axis=0)
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
            remaining_idx = remaining_idx[~confident_mask]

            if self.return_earray:
                ensemble_mask[confident_idx, i] = True
            # endregion
        # endregion

        if self.return_earray:
            return earray(y_score, ensemble_mask)
        else:
            return y_score


_N_CV_THRESHOLDS = 10
_MAX_DEFAULT_CV_THRESHOLD = 0.95


def _fitting_path(estimators, response_method, X, y, sample_weight):
    """Trains cascaded estimators."""
    return fit_one(
        ThresholdCascadeClassifier(
            clone(estimators),
            thresholds=0.0,
            response_method=response_method,
            return_earray=True,
            prefit=False,
            n_jobs=None,
            verbose=False,
        ),
        X,
        y,
        sample_weight=sample_weight,
    )


def _scoring_path(
    cascade,
    thresholds,
    costs,
    X,
    y,
    scoring,
):
    """Trains cascade and scores with accuracy metric."""
    y_pred = cascade.set_params(thresholds=thresholds).predict(X)
    if hasattr(scoring, "_score_func"):
        score = scoring._score_func(y, y_pred)
    else:
        score = scoring(y, y_pred)
    cost = y_pred.acceptance_rates @ costs
    return score, cost


class ThresholdCascadeClassifierCV(ThresholdCascadeClassifier):
    """Cascade of classifiers with Pareto-optimized deferral thresholds.

    Optimizes deferral thresholds via cross-validation grid search, identifying
    non-dominated (Pareto-optimal) threshold configurations that balance accuracy
    and computational cost. Users can select thresholds based on performance
    constraints or cost budgets.

    During inference, runs the first estimator and if a predicted score is lower
    than ``thresholds[0]``, tries the second, and so on. The last estimator always
    makes predictions on deferred samples.

    Parameters
    ----------
    estimators : array-like of object, length n_estimators
        Base estimators. Preferably ordered from weakest (fast, low-accuracy) to
        strongest (slow, high-accuracy).
    costs : array-like of shape (n_estimators,)
        Computational cost per estimator. Used to identify non-dominated
        threshold configurations along the accuracy-cost tradeoff.
    cv_thresholds : array-like of shape (n_thresholds,) or int, default=None
        Candidate deferral thresholds for grid search. If None, defaults to 10
        thresholds linearly spaced from 1/n_classes to 0.95. If int, generates
        that many thresholds in the same range.
    cv : int, cross-validation generator or iterable, default=5
        Cross-validation splitting strategy. Accepts:

        - int: number of folds (uses StratifiedKFold for classification)
        - CV splitter object
        - Iterable yielding (train_idx, test_idx) splits
    scoring : callable or str, default="accuracy"
        Scorer for threshold evaluation. Can be a scikit-learn scorer name
        (e.g., "accuracy", "f1") or a callable with signature
        ``scorer(y_true, y_pred) -> float`` (higher is better).
    min_score : float, default=None
        Minimum acceptable cross-validation score. If specified, selects the
        Pareto config with lowest cost meeting this accuracy constraint.
        If None (default), uses the highest-accuracy Pareto config.
    max_cost : float, default=None
        Maximum acceptable computational cost. If specified, selects the
        Pareto config with highest accuracy within this cost budget.
        If None (default), uses the highest-accuracy Pareto config.
    raise_error : bool, default=False
        Whether to raise ``CascadeParetoConfigException`` if no Pareto configuration
        satisfies the specified constraints (min_score, max_cost). If False (default),
        issues a warning and falls back to the highest-accuracy Pareto config.
    response_method : {"predict_proba", "decision_function"}, default="predict_proba"
        Method by estimators for computing deferral scores.
    return_earray : bool, default=True
        Whether to return :class:`~skfb.core.ENDArray` with ensemble mask
        or plain numpy ndarray.
    prefit : bool, default=False
        If True, estimators are assumed fitted; skips training in ``fit()``.
    n_jobs : int, default=None
        Parallel jobs for CV grid search. -1 uses all processors.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    best_thresholds_ : list of float
        Best selected thresholds.
    all_cv_thresholds_ : ndarray, shape (n_configs, n_splits)
        All generated threshold configurations.
    mean_cv_scores_ : ndarray, shape (n_configs,)
        Average cross-validated classification scores.
    mean_cv_costs_ : ndarray, shape (n_configs,)
        Average cross-validated computational costs.

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.ensemble import ThresholdCascadeClassifierCV
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = np.random.rand(100, 4)
    >>> y = np.random.randint(0, 2, 100)
    >>> cascade_cv = ThresholdCascadeClassifierCV(
    ...     [LogisticRegression(random_state=0),
    ...      RandomForestClassifier(random_state=0)],
    ...     costs=[1.0, 5.0],
    ...     cv_thresholds=5,
    ...     cv=3,
    ... )
    >>> cascade_cv.fit(X, y)
    ThresholdCascadeClassifierCV(...)
    >>> cascade_cv.predict(X[:5])
    array([0, 1, 0, 1, 0])
    >>> # Fit with min accuracy constraint: use cheapest config achieving ≥90%
    >>> cascade_cv_constrained = ThresholdCascadeClassifierCV(
    ...     [LogisticRegression(random_state=0),
    ...      RandomForestClassifier(random_state=0)],
    ...     costs=[1.0, 5.0],
    ...     min_score=0.90,
    ... )
    >>> cascade_cv_constrained.fit(X, y)
    ThresholdCascadeClassifierCV(...)
    >>> cascade_cv_constrained.predict(X[:5])
    array([0, 1, 0, 1, 0])

    Notes
    -----
    The Pareto front contains all non-dominated configurations: those where no
    other configuration achieves both strictly higher score AND strictly lower
    cost.

    If you want a fallback option for the last estimator, consider rejectors
    from :mod:`skfb.estimators`.
    """

    _parameter_constraints = {
        "estimators": ["array-like"],
        "costs": ["array-like"],
        "cv_thresholds": ["array-like", Interval(Real, None, None, closed="neither"), None],
        "cv": ["cv_object"],
        "scoring": [callable, StrOptions(set(get_scorer_names())), None],
        "min_score": [Interval(Real, None, None, closed="neither"), None],
        "max_cost": [Interval(Real, 0, None, closed="left"), None],
        "strategy": [StrOptions({"min_score", "max_cost", "balanced"})],
        "raise_error": ["boolean"],
        "response_method": [StrOptions({"decision_function", "predict_proba"})],
        "return_earray": ["boolean"],
        "prefit": ["boolean"],
        "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        estimators,
        costs,
        cv_thresholds=None,
        min_score=None,
        max_cost=None,
        strategy="min_score",
        cv=5,
        scoring="accuracy",
        raise_error=False,
        response_method="predict_proba",
        return_earray=True,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimators=estimators,
            thresholds=0.0,
            response_method=response_method,
            return_earray=return_earray,
            prefit=False,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.costs = costs
        self.cv_thresholds = cv_thresholds
        self.cv = cv
        self.scoring = scoring
        self.min_score = min_score
        self.max_cost = max_cost
        self.strategy = strategy
        self.raise_error = raise_error

    def _select_best_thresholds(self):
        """Identifies non-dominated (Pareto-optimal) threshold configurations.

        A configuration is Pareto-optimal if no other configuration achieves both
        strictly higher accuracy AND strictly lower cost.
        """
        feasible = np.ones(len(self.all_cv_thresholds_), dtype=bool)
        if self.min_score is not None:
            feasible &= self.mean_cv_scores_ >= self.min_score
        if self.max_cost is not None:
            feasible &= self.mean_cv_costs_ <= self.max_cost

        if not np.any(feasible):
            if self.raise_error:
                raise CascadeParetoConfigException(
                    f"No threshold configuration satisfies constraints: "
                    f"min_score={self.min_score} and max_cost={self.max_cost}."
                )
            else:
                default_idx = np.argmax(self.mean_cv_scores_ / self.mean_cv_costs_)
                self.best_thresholds_ = self.all_cv_thresholds_[default_idx]

                warnings.warn(
                    (
                        f"No threshold configuration satisfies constraints: "
                        f"min_score={self.min_score} and max_cost={self.max_cost}; "
                        f"setting thresholds = {self.best_thresholds_} giving "
                        f"max(scores / costs)."
                    ),
                    category=CascadeParetoConfigWarning,
                )
        else:
            feasible_idx = np.where(feasible)[0]
            feasible_scores = self.mean_cv_scores_[feasible_idx]
            feasible_costs = self.mean_cv_costs_[feasible_idx]

            pareto = np.ones(len(feasible_idx), dtype=bool)

            for i in range(len(feasible_idx)):
                for j in range(len(feasible_idx)):
                    if i != j:
                        j_dominates_i = (
                            feasible_scores[j] >= feasible_scores[i]
                            and feasible_costs[j] <= feasible_costs[i]
                            and (
                                feasible_scores[j] != feasible_scores[i]
                                or feasible_costs[j] != feasible_costs[i]
                            )
                        )
                        if j_dominates_i:
                            pareto[i] = False
                            break

            pareto_idx = feasible_idx[pareto]
            pareto_scores = self.mean_cv_scores_[pareto_idx]
            pareto_costs = self.mean_cv_costs_[pareto_idx]

            if self.strategy == "min_score":
                best_idx = np.argmax(pareto_scores)
            elif self.strategy == "max_cost":
                best_idx = np.argmin(pareto_costs)
            else:
                best_idx = np.argmax(pareto_scores / pareto_costs)

            self.best_thresholds_ = self.all_cv_thresholds_[pareto_idx[best_idx]]

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
        """Fit estimators and identify Pareto-optimal threshold configurations.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, samples are equally weighted.

        Returns
        -------
        self : object
            Fitted estimator. Use ``predict()`` and/or ``set_params()`` methods.
        """
        self.classes_ = unique_labels(y)

        # region Process costs
        if isinstance(self.costs, (tuple, int)):
            self.costs_ = np.array([self.costs] * len(self.estimators))
        else:
            self.costs_ = np.asarray(self.costs)
        # endregion

        # region Generate candidate thresholds
        if self.cv_thresholds is None or isinstance(self.cv_thresholds, int):
            n_thresholds = self.cv_thresholds or _N_CV_THRESHOLDS
            self.cv_thresholds_ = np.linspace(
                1 / len(self.classes_),
                _MAX_DEFAULT_CV_THRESHOLD,
                n_thresholds,
            )
        else:
            self.cv_thresholds_ = np.asarray(self.cv_thresholds)
        # endregion

        # region Setup cross-validation and scorer
        self.cv_ = check_cv(self.cv, y=y, classifier=True)
        self.scoring_ = get_scorer(self.scoring)
        # endregion

        # region Generate all threshold combinations
        threshold_grids = [self.cv_thresholds_] * (len(self.estimators) - 1)
        threshold_combinations = ParameterGrid(
            {
                f"threshold_{i}": thresholds
                for i, thresholds in enumerate(threshold_grids)
            },
        )
        self.all_cv_thresholds_ = [
            tuple(combo[f"threshold_{i}"] for i in range(len(self.estimators) - 1))
            for combo in threshold_combinations
        ]
        self.all_cv_thresholds_ = np.array(self.all_cv_thresholds_)
        # endregion

        # region Temporarily train estimators on different folds
        cascades = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fitting_path)(
                self.estimators,
                self.response_method,
                np.take(X, train_idx, axis=0),
                np.take(y, train_idx, axis=0),
                (
                    np.take(sample_weight, train_idx, axis=0)
                    if sample_weight is not None
                    else None
                ),
            )
            for train_idx, _ in self.cv_.split(X, y)
        )
        cascades = np.array(cascades)
        # endregion

        # region Cross-validation
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_scoring_path)(
                cascade,
                thresholds,
                self.costs_,
                np.take(X, test_idx, axis=0),
                np.take(y, test_idx, axis=0),
                self.scoring_,
            )
            for thresholds in self.all_cv_thresholds_
            for cascade, (_, test_idx) in zip(cascades, self.cv_.split(X, y))
        )
        scores_and_costs = np.array(results).reshape(
            len(self.all_cv_thresholds_), len(cascades), 2
        )
        self.mean_cv_scores_ = scores_and_costs[:, :, 0].mean(axis=1)
        self.mean_cv_costs_ = scores_and_costs[:, :, 1].mean(axis=1)
        # endregion

        # region Set default thresholds and estimators
        self.set_params(min_score=self.min_score, max_cost=self.max_cost)
        super().fit(X, y, sample_weight=sample_weight)
        # endregion

        return self

    def set_params(self, **params):
        """Sets the parameters of the cascade.

        If thresholds or new constraints are provided, the transformations are done
        accordingly, so there is no need to refit the cascade.

        Parameters
        ----------
        **params : dict
            Cascade parameters.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If all `min_score`, `max_cost`, and `thresholds` are passed.
        """
        not_given = "not-given"
        max_cost = params.get("max_cost", not_given)
        min_score = params.get("min_score", not_given)
        thresholds = params.get("thresholds", not_given)

        if all(p != not_given for p in (max_cost, min_score, thresholds)):
            raise ValueError(
                "Pass either min_score and max_cost or thresholds. "
                "The former will automatically determine the best thresholds."
            )

        elif thresholds != not_given:
            self._set_thresholds(thresholds)
            params.pop("thresholds")

        elif max_cost != not_given or min_score != not_given:
            self._select_best_thresholds()
            self._set_thresholds(self.best_thresholds_)

        return super().set_params(**params)
