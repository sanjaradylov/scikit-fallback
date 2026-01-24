"""Threshold-based cascade ensembles."""

from typing import Sequence

import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted, clone
from sklearn.metrics import accuracy_score, get_scorer, get_scorer_names
from sklearn.model_selection import check_cv, ParameterGrid
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


def _cost_aware_score(y_true, y_pred, scorer, costs, cost_weight=0.0):
    """Compute cost-aware score balancing accuracy and computational costs.

    Follows the Cascadia approach: combines accuracy score with accumulated
    computational costs of using different estimators in the cascade.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : FBNDArray or ndarray
        Predictions with optional ensemble mask.
    scorer : callable or sklearn scorer
        Scorer object. Can be sklearn scorer (which has _score_func method)
        or a plain callable that takes (y_true, y_pred).
    costs : array-like
        Cost per estimator (includes all estimators).
    cost_weight : float, default=0.0
        Weight for the cost term. Higher values penalize computational cost more.

    Returns
    -------
    score : float
        Cost-aware score.
    """
    # Get base accuracy score using the scorer
    # If it's a sklearn scorer, use _score_func; otherwise call directly.
    if hasattr(scorer, "_score_func"):
        accuracy = scorer._score_func(y_true, y_pred)
    else:
        accuracy = scorer(y_true, y_pred)

    # If cost_weight is 0 or y_pred is not an array with ensemble_mask,
    # return accuracy only.
    if cost_weight == 0.0 or not hasattr(y_pred, "ensemble_mask"):
        return accuracy

    # Calculate total cost from ensemble mask
    # ensemble_mask shape: (n_samples, n_estimators).
    n_samples = y_pred.ensemble_mask.shape[0]
    ensemble_costs = y_pred.ensemble_mask @ np.array(costs)  # Cost per sample
    total_cost = ensemble_costs.sum() / n_samples  # Average cost

    # Return combined score: accuracy - cost_weight * avg_cost
    return accuracy - cost_weight * total_cost


def _scoring_path(
    estimators,
    costs,
    thresholds,
    X_train,
    X_test,
    y_train,
    y_test,
    scorer,
    response_method,
    cost_weight=0.0,
):
    """Trains cascade and scores with cost-aware metric.

    Parameters
    ----------
    estimators : list
        Base estimators to train.
    costs : array-like
        Cost per estimator.
    thresholds : array-like or float
        Deferral thresholds for cascade.
    X_train, X_test : array-like
        Training and test features.
    y_train, y_test : array-like
        Training and test labels.
    scorer : callable
        Scorer function.
    response_method : str
        Method for confidence scores.
    cost_weight : float, default=0.0
        Weight for computational cost in the objective.

    Returns
    -------
    score : float
        Cost-aware score on test set.
    """
    cascade = ThresholdCascadeClassifier(
        clone(estimators),
        thresholds=thresholds,
        response_method=response_method,
        return_earray=True,
        prefit=False,
        n_jobs=None,
        verbose=False,
    )
    y_pred = cascade.fit(X_train, y_train).predict(X_test)
    score = _cost_aware_score(y_test, y_pred, scorer, costs, cost_weight)
    return score


class ThresholdCascadeClassifierCV(ThresholdCascadeClassifier):
    """Cascade of classifiers w/ deferrals based on tuned thresholds.

    Optimizes deferral thresholds via cross-validation and grid search to balance
    classification accuracy with computational costs.
    During inference, runs the first estimator and if a predicted score is lower than
    ``thresholds[0]``, tries the second, and so on. The last estimator always makes
    predictions on the samples deferred by the previous estimators.

    Parameters
    ----------
    estimators : array-like of object, length n_estimators
        Base estimators. Preferrably, from weakest (e.g., rule-based or linear) to
        strongest (e.g., gradient boosting).
    costs : array-like of shape (n_estimators,)
        Computational cost per estimator. Used to optimize thresholds balancing
        accuracy and computational efficiency.
    cv_thresholds : array-like of shape (n_thresholds,) or int, default=None
        Array of candidate deferral thresholds to evaluate via grid search.
        If None, defaults to 10 thresholds linearly spaced from 1/n_classes to 0.95.
        If int, generates that many thresholds in the same range.
    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the efficient Leave-One-Out cross-validation
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
    scoring : callable or str, default="accuracy"
        A scorer callable object or scikit-learn scorer name.
        Should accept (y_true, y_pred) and return a higher-is-better score.
        Common choices: "accuracy", "f1", "precision", "recall".
        If custom callable, signature must be scoring(y_true, y_pred) -> float.
    cost_weight : float, default=0.0
        Weight for the computational cost term in the objective:
        objective = accuracy - cost_weight * (total_cost / n_samples).
        Higher values penalize computational cost more heavily.
        If 0.0 (default), only accuracy is optimized.
    response_method : {"predict_proba", "decision_function"}, default="predict_proba"
        Method by estimators for deferral score.
        For ``"decision_function"``, ``thresholds`` can be negative.
    return_earray : bool, default=True
        Whether to return :class:`~skfb.core.ENDArray` of predicted classes
        or plain numpy ndarray.
    prefit : bool, default=False
        Whether estimators are fitted. If True, checks their ``classes_`` attributes
        for intercompatibility and skips fitting in ``fit()``.
    n_jobs : int, default=None
        Number of parallel jobs for CV grid search. -1 uses all processors.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    best_thresholds_ : list of float
        Optimal deferral thresholds selected via CV grid search.
    cv_results_ : dict
        Cross-validation results with keys:

        - ``mean_scores`` : array of shape (n_param_combinations,)
          Mean test score for each threshold configuration.
        - ``std_scores`` : array of shape (n_param_combinations,)
          Standard deviation of test scores.
        - ``param_list`` : list of threshold configurations tried.

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.ensemble import ThresholdCascadeClassifierCV
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = np.random.rand(100, 4)
    >>> y = np.random.randint(0, 2, 100)
    >>> maxent = LogisticRegression(random_state=0)
    >>> rf = RandomForestClassifier(random_state=0, n_estimators=10)
    >>> costs = [1.0, 5.0]  # Cascade: fast LR then slow RF
    >>> cascade_cv = ThresholdCascadeClassifierCV(
    ...     [maxent, rf],
    ...     costs=costs,
    ...     cv_thresholds=5,
    ...     cv=3,
    ...     cost_weight=0.1,
    ... )
    >>> cascade_cv.fit(X, y)
    ThresholdCascadeClassifierCV(...)
    >>> cascade_cv.predict(X[:5])
    array([0, 1, 0, 1, 0])

    Notes
    -----
    The threshold grid search explores all combinations of candidate thresholds
    across estimators (cartesian product) and selects the combination with the
    highest cost-aware score via cross-validation.

    If you want a fallback option (for the last estimator), consider rejectors
    from :mod:`skfb.estimators`.
    """

    _parameter_constraints = {
        "estimators": ["array-like"],
        "costs": ["array-like"],
        "cv_thresholds": ["array-like", Interval(Real, None, None, closed="neither"), None],
        "cv": ["cv_object"],
        "scoring": [callable, StrOptions(set(get_scorer_names())), None],
        "cost_weight": [Interval(Real, 0, None, closed="left")],
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
        cv=5,
        scoring="accuracy",
        cost_weight=0.0,
        response_method="predict_proba",
        return_earray=True,
        prefit=False,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimators=estimators,
            thresholds=0.0,
            response_method=response_method,
            return_earray=return_earray,
            prefit=prefit,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.costs = costs
        self.cv_thresholds = cv_thresholds
        self.cv = cv
        self.scoring = scoring
        self.cost_weight = cost_weight

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
        """Fit estimators and optimize thresholds via cross-validation.

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

        # region Generate candidate thresholds if needed
        if self.cv_thresholds is None or isinstance(self.cv_thresholds, int):
            n_thresholds = self.cv_thresholds or _N_CV_THRESHOLDS
            cv_thresholds = np.linspace(
                1 / len(self.classes_),
                _MAX_DEFAULT_CV_THRESHOLD,
                n_thresholds,
            )
        else:
            cv_thresholds = np.asarray(self.cv_thresholds)

        self.cv_thresholds_ = cv_thresholds
        # endregion

        # region Fit base estimators if not prefit
        if not hasattr(self, "estimators_"):
            self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(fit_one)(estimator, X, y, sample_weight)
                for estimator in self.estimators
            )
        # endregion

        # region Set up cross-validation and scorer
        cv = check_cv(self.cv, y=y, classifier=True)
        scorer = get_scorer(self.scoring)
        # endregion

        # region Generate all threshold combinations (grid search)
        n_estimators = len(self.estimators_)

        # Create all combinations of thresholds for n_estimators - 1 positions
        # (last estimator doesn't have a threshold)
        threshold_grids = [cv_thresholds] * (n_estimators - 1)
        threshold_combinations = ParameterGrid(
            {
                f"threshold_{i}": thresholds
                for i, thresholds in enumerate(threshold_grids)
            },
        )

        # Convert to list of threshold tuples for _scoring_path
        threshold_configs = [
            tuple(combo[f"threshold_{i}"] for i in range(n_estimators - 1))
            for combo in threshold_combinations
        ]
        # endregion

        # region Run cross-validation scoring for all threshold combinations
        all_scores = []

        for threshold_config in threshold_configs:
            cv_scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_scoring_path)(
                    self.estimators,
                    self.costs,
                    threshold_config,
                    X[train_idx],
                    X[test_idx],
                    y[train_idx],
                    y[test_idx],
                    scorer,
                    self.response_method,
                    self.cost_weight,
                )
                for train_idx, test_idx in cv.split(X, y)
            )
            all_scores.append(cv_scores)
        # endregion

        # region Select best threshold configuration
        all_scores = np.array(all_scores)  # Shape: (n_configs, n_splits)
        best_idx = all_scores.mean(axis=1).argmax()
        self.best_thresholds_ = list(threshold_configs[best_idx])
        self.cv_scores_ = all_scores
        # endregion

        # region Set optimal thresholds and current estimators
        self._set_thresholds(self.best_thresholds_)
        self._current_estimators = self.estimators_[:]
        self._current_thresholds = self.thresholds_[:]
        # endregion

        self.is_fitted_ = True

        return self
