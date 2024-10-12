"""Classification w/ fallback option based on class-specific thresholds."""

__all__ = (
    "multi_threshold_predict_or_fallback",
    "MultiThresholdFallbackClassifier",
)

import warnings

import numpy as np

from scipy.stats import uniform

from sklearn.base import clone
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import check_cv, ParameterSampler

try:
    from sklearn.utils.parallel import delayed, Parallel
except ModuleNotFoundError:
    from joblib import Parallel

    # pylint: disable=ungrouped-imports
    from sklearn.utils.fixes import delayed

    warnings.warn(
        "Using ``joblib.Parallel`` for ``MultiTresholdFallbackClassifierCV`` instead "
        "of ``sklearn.utils.parallel.Parallel``, which was added in sklearn 1.3.",
        category=ImportWarning,
    )

from sklearn.utils.validation import check_is_fitted, NotFittedError

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
        "fallback_mode": [StrOptions({"return", "store", "ignore"})],
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
    fallback_mode : {"store", "return", "ignore"}, default="return"
        Whether to have:

        * (``"return"``) a numpy ndarray of both predictions and fallbacks;
        * (``"store"``)  an FBNDArray of predictions storing also fallback mask;
        * (``"ignore"``) a numpy ndarray of only estimator's predictions.
    return_probas : bool, default=False
        Whether to return also probabilities.

    Returns
    -------
    FBNDArray or ndarray, or tuple of FBNDArray of shape = n_samples \
            and ndarray of shape (n_samples, n_classes)
        Predictions w/o fallbacks, combined predictions, or predictions w/
        fallback mask; depends on ``fallback_mode`` and ``return_probas``.

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
    elif fallback_mode == "ignore":
        return classes.take(y_prob.argmax(axis=1))
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
    fallback_mode : {"return", "store", "ignore"}, default="store"
        While predicting, whether to return:

        * (``"return"``) a numpy ndarray of both predictions and fallbacks;
        * (``"store"``)  an FBNDArray of predictions storing also fallback mask;
        * (``"ignore"``) a numpy ndarray of only estimator's predictions.

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


MAX_THRESHOLD = 1.0


def yield_thresholds(classes, n_iter=10, random_state=None):
    """Generates local thresholds either exhaustively or randomly."""
    min_threshold = 1 / len(classes)
    max_threshold = MAX_THRESHOLD

    distributions = {
        c: uniform(loc=min_threshold, scale=max_threshold - min_threshold)
        for c in classes
    }
    sampler = ParameterSampler(distributions, n_iter=n_iter, random_state=random_state)

    return (tuple(s.values()) for s in sampler)


def _generate_scoring_path(
    threshold_grid,
    cv,
    base_estimator,
    X,
    y,
    scoring,
    fallback_label,
    fallback_mode,
):
    """Fits estimator every cv iteration, yields thresholds, and runs threshold eval."""
    for train_idx, test_idx in cv.split(X, y):
        estimator = clone(base_estimator).fit(X[train_idx], y[train_idx])
        for thresholds in threshold_grid:
            yield delayed(_scoring_path)(
                estimator,
                thresholds,
                X[train_idx],
                X[test_idx],
                y[train_idx],
                y[test_idx],
                scoring,
                fallback_label,
                fallback_mode,
            )


def _scoring_path(
    base_estimator,
    thresholds,
    X_train,
    X_test,
    y_train,
    y_test,
    scoring,
    fallback_label=-1,
    fallback_mode="store",
):
    """Trains and scores an estimator w/ a reject option."""
    estimator = MultiThresholdFallbackClassifier(
        base_estimator,
        thresholds=thresholds,
        ambiguity_threshold=0.0,
        fallback_label=fallback_label,
        fallback_mode=fallback_mode,
    )
    y_pred = estimator.fit(X_train, y_train).predict(X_test)
    return scoring(y_test, y_pred)


class MultiThresholdFallbackClassifierCV(MultiThresholdFallbackClassifier):
    """A fallback classifier based on local thresholds tuned with cross-validation.

    Uniformly samples local thresholds ranging from the maximum threshold s.t. no
    samples are rejected and 1.0. After sampling, evaluates performance w/ cv and
    chooses the best thresholds w.r.t scoring.

    Parameters
    ----------
    estimator : object
        Fitted base estimator supporting probabilistic predictions.
    ambiguity_threshold : float, default=0.0
        Predictions w/ the close top 2 scores are rejected
    n_iter : int, default=10
        Number of thresholds to generate.
    random_state : int or numpy.random.RandomState, default=None
        Random state.
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
        Defaults to either ``skfb.metrics.prediction_quality(accuracy_score)``
        for ``fallback_mode="return"`` or ``skfb.metrics.predict_reject_accuracy_score``
        for ``fallback_mode="store``.
    n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
    verbose : int, default=0
        Verbosity level.
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
    >>> from skfb.estimators import MultiThresholdFallbackClassifierCV
    >>> X = np.array(
    ...     [
    ...         [-3, -3], [-3, -2], [-2, -3],
    ...         [2, 3], [2, 4], [3, 4], [3.1, 3], [3, 2.9],
    ...         [4, 3], [3, 2], [4, 2], [2.9, 3], [3, 3.1],
    ...     ])
    >>> y = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 5)
    >>> estimator = LogisticRegression(C=10_000, random_state=0)
    >>> rejector = MultiThresholdFallbackClassifierCV(
    ...     estimator=estimator, cv=2, n_iter=10, random_state=0).fit(X, y)

    Notes
    -----
    Supports also binary classifiers.
    """

    _parameter_constraints = {**MultiThresholdFallbackClassifier._parameter_constraints}
    _parameter_constraints.pop("thresholds")
    _parameter_constraints.update(
        {
            "n_iter": [Interval(Integral, 1, None, closed="left"), None],
            "cv": ["cv_object"],
            "scoring": [callable, StrOptions(set(get_scorer_names())), None],
            "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
            "verbose": [bool, int],
        },
    )

    def __init__(
        self,
        estimator,
        *,
        ambiguity_threshold=0.0,
        n_iter=10,
        cv=None,
        scoring=None,
        random_state=None,
        n_jobs=None,
        verbose=0,
        fallback_label=-1,
        fallback_mode="store",
    ):
        super().__init__(
            estimator=estimator,
            thresholds={},
            ambiguity_threshold=ambiguity_threshold,
            fallback_label=fallback_label,
            fallback_mode=fallback_mode,
        )

        del self.thresholds

        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, X, y, set_attributes=None, **fit_params):
        """Fits the base estimator and finds the best threshold."""
        set_attributes = set_attributes or {}

        # region Create threshold grid
        classes_ = set_attributes.get("classes_")
        threshold_grid = list(
            yield_thresholds(
                classes_,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        )
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
            _generate_scoring_path(
                threshold_grid,
                cv_,
                self.estimator,
                X,
                y,
                scoring_,
                set_attributes.get("fallback_label_", self.fallback_label),
                self.fallback_mode,
            )
        )
        cv_scores_ = np.array(scores).reshape(cv_.n_splits, -1)
        # endregion

        # region Update fitted attributes
        mean_cv_scores = cv_scores_.mean(axis=1)
        thresholds_ = threshold_grid[np.argmax(mean_cv_scores)]
        best_score_ = mean_cv_scores.max()
        set_attributes.update(
            {
                "scoring_": scoring_,
                "thresholds_": thresholds_,
                "cv_": cv_,
                "cv_scores_": cv_scores_,
                "best_score_": best_score_,
            },
        )
        set_attributes.update(
            super()._fit(X, y, set_attributes=set_attributes, **fit_params)
        )
        # endregion

        return set_attributes

    def _predict(self, X):
        """Predicts classes based on the learned local thresholds."""
        return multi_threshold_predict_or_fallback(
            self.estimator_,
            X,
            classes=self.classes_,
            thresholds=self.thresholds_,
            ambiguity_threshold=self.ambiguity_threshold,
            fallback_label=self.fallback_label_,
            fallback_mode=self.fallback_mode,
        )

    def _set_fallback_mask(self, y_prob):
        """Sets the fallback mask for predicted probabilities."""
        y_prob.fallback_mask = _is_top_low(y_prob, self.thresholds_) | _are_top_2_close(
            y_prob, self.ambiguity_threshold
        )
