"""Classification w/ fallback option based on decision threshold."""

import warnings

from sklearn.base import clone
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import check_cv

try:
    from sklearn.utils.parallel import delayed, Parallel
except ModuleNotFoundError:
    from joblib import Parallel

    # pylint: disable=ungrouped-imports
    from sklearn.utils.fixes import delayed

    warnings.warn(
        "Using ``joblib.Parallel`` for ``TresholdFallbackClassifierCV`` and "
        "``RateFallbackClassifierCV`` instead of ``sklearn.utils.parallel.Parallel``, "
        "which was added in sklearn 1.3.",
        category=ImportWarning,
    )

from sklearn.utils.validation import check_is_fitted, NotFittedError

import numpy as np

from .base import BaseFallbackClassifier
from ..core import array as ska
from ..metrics._classification import get_scoring
from ..utils._legacy import (
    HasMethods,
    Integral,
    Interval,
    Real,
    StrOptions,
    validate_params,
)


def _top_2(scores):
    """Returns top 2 scores."""
    return np.sort(scores)[-2:]


def _are_top_2_close(y_score, ambiguity_threshold):
    """Whether the scores of top two classes are lower than the ambiguity threshold."""
    y_top_2 = np.apply_along_axis(_top_2, 1, y_score)
    y_diff = np.apply_over_axes(np.diff, y_top_2, 1)
    return y_diff[:, 0] < ambiguity_threshold


def _is_top_low(y_score, threshold):
    """Whether the top confidence score is lower than the threshold."""
    return y_score.max(axis=1) < threshold


@validate_params(
    {
        "estimator": [HasMethods(["fit", "predict_proba"])],
        "X": ["array-like", "sparse matrix"],
        "classes": [np.ndarray],
        "threshold": [Interval(Real, 0.0, 1.0, closed="both")],
        "fallback_mode": [StrOptions({"return", "store", "ignore"})],
    },
    prefer_skip_nested_validation=True,
)
def predict_or_fallback(
    estimator,
    X,
    classes,
    threshold=0.5,
    ambiguity_threshold=0.0,
    fallback_label=-1,
    fallback_mode="return",
):
    """For every sample, either predicts a class or rejects a prediction.

    Parameters
    ----------
    estimator : object
        Fitted base estimator supporting probabilistic predictions.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input samples.
    classes : ndarray, shape = (n_classes,)
        NDArray of class labels.
    threshold : float, default=0.5
        Predictions w/ the lower thresholds are rejected.
    ambiguity_threshold : float, default=0.0
        Predictions w/ the close top 2 scores are rejected.
    fallback_label : any, default=-1
        Rejected samples are labeled w/ this label.
    fallback_mode : {"store", "return", "ignore"}, default="return"
        Whether to have:

        * (``"return"``) a numpy ndarray of both predictions and fallbacks;
        * (``"store"``)  an FBNDArray of predictions storing also fallback mask;
        * (``"ignore"``) a numpy ndarray of only estimator's predictions.

    Returns
    -------
    FBNDArray or ndarray, shape (n_samples,)
        Predictions w/o fallbacks, combined predictions, or predictions w/
        fallback mask; depends on ``fallback_mode``.
    """
    y_prob = estimator.predict_proba(X)
    fallback_mask = _is_top_low(y_prob, threshold)
    if ambiguity_threshold > 0.0:
        fallback_mask |= _are_top_2_close(y_prob, ambiguity_threshold)

    if fallback_mode == "return":
        acceptance_mask = ~fallback_mask
        y_comb = np.empty(len(fallback_mask), dtype=classes.dtype)
        y_comb[acceptance_mask] = classes.take(y_prob[acceptance_mask].argmax(axis=1))
        y_comb[fallback_mask] = fallback_label
        return y_comb
    elif fallback_mode == "ignore":
        return y_prob.argmax(axis=1)
    else:
        y_pred = classes.take(y_prob.argmax(axis=1))
        y_pred = ska.fbarray(y_pred, fallback_mask)
        return y_pred


class ThresholdFallbackClassifier(BaseFallbackClassifier):
    """A fallback classifier based on provided certainty threshold.

    Parameters
    ----------
    estimator : object
        The base estimator making decisions w/o fallbacks.
    threshold : float, default=0.5
        The fallback threshold.
    ambiguity_threshold : float, default=0.0
        Predictions w/ the close top 2 scores are rejected.
    fallback_label : any, default=-1
        The label of a rejected example.
        Should be compatible w/ the class labels from training data.
    fallback_mode : {"return", "store", "ignore"}, default="store"
        While predicting, whether to return:

        * (``"return"``) a numpy ndarray of both predictions and fallbacks;
        * (``"store"``)  an FBNDArray of predictions storing also fallback mask;
        * (``"ignore"``) a numpy ndarray of only estimator's predictions.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skfb.estimators import ThresholdFallbackClassifier
    >>> X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2., 2.5]])
    >>> y = np.array([0, 1, 0, 1, 0, 1])
    >>> estimator = LogisticRegression(random_state=0)
    >>> rejector = ThresholdFallbackClassifier(estimator, threshold=0.6).fit(X, y)
    >>> y_pred = rejector.predict(X)
    >>> y_pred, y_pred.get_dense_fallback_mask()
    (FBNDArray([0, 1, 0, 1, 1, 1]),
     array([False, False, False, False,  True, False]))
    >>> rejector.set_params(fallback_mode="return").predict(X)
    array([ 0,  1,  0,  1, -1,  1])
    >>> rejector.score(X, y)
    1.0
    >>> rejector.set_params(fallback_mode="store").score(X, y)
    0.8333333333333334
    """

    _parameter_constraints = {**BaseFallbackClassifier._parameter_constraints}
    _parameter_constraints.update(
        {
            "threshold": [Interval(Real, 0.0, 1.0, closed="both")],
            "ambiguity_threshold": [Interval(Real, 0.0, 1.0, closed="both")],
        },
    )

    def __init__(
        self,
        estimator,
        *,
        threshold=0.5,
        ambiguity_threshold=0.0,
        fallback_label=-1,
        fallback_mode="store",
    ):
        super().__init__(
            estimator=estimator,
            fallback_label=fallback_label,
            fallback_mode=fallback_mode,
        )
        self.threshold = threshold
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
        """Predicts classes based on the fixed certainty threshold."""
        return predict_or_fallback(
            self.estimator_,
            X,
            self.classes_,
            threshold=self.threshold,
            ambiguity_threshold=self.ambiguity_threshold,
            fallback_label=self.fallback_label_,
            fallback_mode=self.fallback_mode,
        )

    def _set_fallback_mask(self, y_prob, X=None):
        """Sets the fallback mask for predicted probabilites."""
        y_prob.fallback_mask = _is_top_low(y_prob, self.threshold) | _are_top_2_close(
            y_prob, self.ambiguity_threshold
        )


def _scoring_path(
    base_estimator,
    threshold,
    X_train,
    X_test,
    y_train,
    y_test,
    scoring,
    fallback_label=-1,
    fallback_mode="store",
):
    """Trains and scores an estimator w/ a reject option."""
    estimator = ThresholdFallbackClassifier(
        clone(base_estimator),
        threshold=threshold,
        ambiguity_threshold=0.0,
        fallback_label=fallback_label,
        fallback_mode=fallback_mode,
    )
    y_pred = estimator.fit(X_train, y_train).predict(X_test)
    return scoring(y_test, y_pred)


_N_THRESHOLDS = 10


class ThresholdFallbackClassifierCV(ThresholdFallbackClassifier):
    """A fallback classifier based on the best certainty threshold learnt via CV.

    Parameters
    ----------
    estimator : object
        The base estimator making decisions.
    thresholds : array-like of shape (n_thresholds,) or int, default=None
        Array of fallback thresholds to evaluate.
        If None, defaults to 10 thresholds from p = 1 / len(classes),
        which is about not falling back, to 0.95. Same with int except that the number
        of threshold equals this value.
    ambiguity_threshold : float, default=0.0
        Predictions w/ the close top 2 scores are rejected.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the efficient Leave-One-Out cross-validation
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
    scoring : callable or str, default=None
        A scorer callable object supporting a reject option, such as metrics from
        :mod:`~skfb.metrics`.
        If not from scikit-learn, make sure to wrap it with ``make_scorer``.
    verbose : int, default=0
        Verbosity level .
    fallback_label : any, default=-1
        The label of a rejected example.
        Should be compatible w/ the class labels from training data.
    fallback_mode : {"return", "store", "ignore"}, default="store"
        While predicting, whether to return:

        * (``"return"``) a numpy ndarray of both predictions and fallbacks;
        * (``"store"``)  an FBNDArray of predictions storing also fallback mask;
        * (``"ignore"``) a numpy ndarray of only estimator's predictions.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skfb.estimators import ThresholdFallbackClassifierCV
    >>> X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2., 2.5]])
    >>> y = np.array([0, 1, 0, 1, 0, 1])
    >>> estimator = LogisticRegression(random_state=0)
    >>> rejector = ThresholdFallbackClassifierCV(
    ...     estimator=estimator,
    ...     thresholds=(0.5, 0.55, 0.6, 0.65),
    ...     cv=2)
    >>> rejector.fit(X, y)
    ThresholdFallbackClassifierCV(cv=2,
                                  estimator=LogisticRegression(random_state=0),
                                  n_jobs=2, thresholds=(0.5, 0.55, 0.6, 0.65),
                                  verbose=1)
    >>> rejector.threshold_
    0.55
    >>> rejector.best_score_
    0.8333333333333333
    >>> y_pred = rejector.predict(X)
    >>> y_pred, y_pred.get_dense_fallback_mask()
    (FBNDArray([0, 1, 0, 1, 1, 1]),
     array([False, False, False, False,  True, False]))
    >>> rejector.set_params(fallback_mode="return").predict(X)
    array([ 0,  1,  0,  1, -1,  1])
    >>> rejector.score(X, y)
    1.0
    >>> rejector.set_params(fallback_mode="store").score(X, y)
    1.0
    """

    _parameter_constraints = {**ThresholdFallbackClassifier._parameter_constraints}
    _parameter_constraints.pop("threshold")
    _parameter_constraints.update(
        {
            "thresholds": ["array-like", int, None],
            "cv": ["cv_object"],
            "scoring": [callable, StrOptions(set(get_scorer_names())), None],
            "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
        },
    )

    def __init__(
        self,
        estimator,
        *,
        thresholds=None,
        ambiguity_threshold=0.0,
        cv=None,
        scoring=None,
        n_jobs=None,
        verbose=0,
        fallback_label=-1,
        fallback_mode="store",
    ):
        super().__init__(
            estimator=estimator,
            ambiguity_threshold=ambiguity_threshold,
            fallback_label=fallback_label,
            fallback_mode=fallback_mode,
        )

        del self.threshold

        self.thresholds = thresholds
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, X, y, set_attributes=None, **fit_params):
        """Fits the base estimator and finds the best threshold."""
        set_attributes = set_attributes or {}

        # region Maybe create thresholds.
        if self.thresholds is None or isinstance(self.thresholds, int):
            classes = set_attributes.get("classes_")
            n_thresholds = self.thresholds or _N_THRESHOLDS
            if classes is None:
                thresholds_ = np.linspace(0.5, 0.95, n_thresholds)
            else:
                thresholds_ = np.linspace(1 / len(classes), 0.95, n_thresholds)
        else:
            thresholds_ = self.thresholds
        # endregion

        # region Validate and/or create objects for cv.
        cv_ = check_cv(self.cv, y=y, classifier=True)
        # endregion

        # region Validate and/or create scoring.
        scoring_ = get_scoring(
            scoring=self.scoring,
            fallback_label=set_attributes.get(
                "fallback_label_",
                self.fallback_label,
            ),
            fallback_mode=self.fallback_mode,
        )
        # endregion

        # region Run maybe parallel scoring paths
        scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="processes")(
            delayed(_scoring_path)(
                self.estimator,
                threshold,
                X[train_idx],
                X[test_idx],
                y[train_idx],
                y[test_idx],
                scoring_,
                fallback_label=set_attributes.get(
                    "fallback_label_",
                    self.fallback_label,
                ),
                fallback_mode=self.fallback_mode,
            )
            for threshold in thresholds_
            for train_idx, test_idx in cv_.split(X, y)
        )
        cv_scores_ = np.array(scores).reshape(len(thresholds_), cv_.n_splits)
        # endregion

        # region Update fitted attributes
        mean_cv_scores = cv_scores_.mean(axis=1)
        threshold_ = thresholds_[np.argmax(mean_cv_scores)]
        best_score_ = mean_cv_scores.max()
        set_attributes.update(
            {
                "thresholds_": thresholds_,
                "cv_": cv_,
                "cv_scores_": cv_scores_,
                "scoring_": scoring_,
                "threshold_": threshold_,
                "best_score_": best_score_,
            },
        )
        set_attributes.update(
            super()._fit(X, y, set_attributes=set_attributes, **fit_params)
        )
        # endregion

        return set_attributes

    def _predict(self, X):
        """Predicts classes based on the learned certainty threshold."""
        return predict_or_fallback(
            self.estimator_,
            X,
            self.classes_,
            threshold=self.threshold_,
            fallback_label=self.fallback_label_,
            fallback_mode=self.fallback_mode,
        )

    def _set_fallback_mask(self, y_prob, X=None):
        """Sets the fallback mask for predicted probabilites."""
        y_prob.fallback_mask = _is_top_low(y_prob, self.threshold_) | _are_top_2_close(
            y_prob, self.ambiguity_threshold
        )


def _find_threshold(estimator, X_train, X_test, y_train, fallback_rate):
    """Finds threshold t s.t. P(estimator(X_test) < t) = fallback_rate."""
    y_prob = estimator.fit(X_train, y_train).predict_proba(X_test).max(axis=1)
    return np.quantile(y_prob, fallback_rate)


class RateFallbackClassifierCV(BaseFallbackClassifier):
    """Fallback classifier learning a threshold based on the provided fallback rate.

    The threshold is determined during training via cross-validation and equals to the
    mean fallback-rate quantile of the predictions on the validation sets.
    Then predictions w/ probabilities lower than the learned threshold are rejected.

    Parameters
    ----------
    estimator : object
        The base estimator supporting probabilistic predictions.
    fallback_rates : array-like of shape (n_fallback_rates,) or float, default=0.05
        The rate(s) of rejected test samples.

        .. deprecated:: 0.1
            ``fallback_rates`` was deprecated in 0.1 and will be removed in 0.2.
            Use ``fallback_rate`` instead.
    fallback_rate : float, default=0.1
        The rate of rejected test samples.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the efficient Leave-One-Out cross-validation
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
    verbose : int, default=0
        Verbosity level.
    fallback_label : any, default=-1
        The label of a rejected example.
        Should be compatible w/ the class labels from training data.
    fallback_mode : {"return", "store"}, default="store"
        While predicting, whether to return:

        * (``"return"``) a numpy ndarray of both predictions and fallbacks;
        * (``"store"``)  an FBNDArray of predictions storing also fallback mask;
        * (``"ignore"``) a numpy ndarray of only estimator's predictions.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skfb.estimators import RateFallbackClassifierCV
    >>> estimator = LogisticRegression(random_state=0)
    >>> X_accept = [[0, 0], [6, 6], [0, 1], [5, 6], [1, 1], [5, 5], [1, 0], [6, 5]]
    >>> X_ambiguous = [[3.25, 3], [3., 3.25]]
    >>> X = np.array(X_accept + X_ambiguous)
    >>> y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    >>> rejector = RateFallbackClassifierCV(
    ...     estimator,
    ...     fallback_rate=0.2,
    ...     cv=2,
    ...     fallback_label=-1,
    ...     fallback_mode="return")
    >>> rejector.fit(X, y).score(X, y)  # Both ambiguous samples were rejected.
    1.0
    >>> rejector.set_params(fallback_mode="store").score(X, y)
    0.9
    """

    _parameter_constraints = {**BaseFallbackClassifier._parameter_constraints}
    _parameter_constraints.update(
        {
            "fallback_rate": [Interval(Real, 0.0, 1.0, closed="both")],
            "cv": ["cv_object"],
            "scoring": [callable, None],
            "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
        },
    )

    def __init__(
        self,
        estimator,
        *,
        fallback_rate=0.1,
        fallback_rates=None,
        cv=None,
        n_jobs=None,
        verbose=0,
        fallback_label=-1,
        fallback_mode="store",
    ):
        super().__init__(
            estimator,
            fallback_label=fallback_label,
            fallback_mode=fallback_mode,
        )

        if fallback_rates is not None:
            warnings.warn(
                "`fallback_rates` was deprecated in version 0.1 and will be removed "
                "in 0.2. Use `fallback_rate` instead (see "
                "https://github.com/sanjaradylov/scikit-fallback/issues/11)",
                category=FutureWarning,
            )

        self.fallback_rate = fallback_rate
        self.fallback_rates = fallback_rates
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, X, y, set_attributes=None, **fit_params):
        """Learns the best threshold."""
        set_attributes = set_attributes or {}

        cv_ = check_cv(self.cv, y=y, classifier=True)

        estimator = clone(self.estimator)

        thresholds_ = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, prefer="processes"
        )(
            delayed(_find_threshold)(
                estimator,
                X[train_idx],
                X[test_idx],
                y[train_idx],
                self.fallback_rate,
            )
            for train_idx, test_idx in cv_.split(X, y)
        )

        set_attributes.update(
            {
                "cv_": cv_,
                "thresholds_": np.array(thresholds_),
                "threshold_": np.mean(thresholds_),
            },
        )
        set_attributes.update(super()._fit(X, y, set_attributes, **fit_params))

        return set_attributes

    def _predict(self, X):
        """Predicts classes based on the learned certainty threshold."""
        return predict_or_fallback(
            self.estimator_,
            X,
            self.classes_,
            threshold=self.threshold_,
            fallback_label=self.fallback_label_,
            fallback_mode=self.fallback_mode,
        )

    def _set_fallback_mask(self, y_prob, X=None):
        """Sets the fallback mask for predicted probabilites."""
        y_prob.fallback_mask = y_prob.max(axis=1) < self.threshold_
