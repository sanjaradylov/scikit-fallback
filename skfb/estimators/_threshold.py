"""Classification w/ fallback option based on decision threshold."""

import abc
import warnings

# pylint: disable=no-name-in-module
# pyright: reportAttributeAccessIssue=false
from sklearn.base import _fit_context, BaseEstimator, MetaEstimatorMixin
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import check_cv

# pylint: disable=import-error,no-name-in-module
# pyright: reportMissingModuleSource=false
from sklearn.utils._param_validation import (
    HasMethods,
    Integral,
    Interval,
    Real,
    StrOptions,
    validate_params,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.parallel import delayed, Parallel
from sklearn.utils.validation import check_X_y, check_is_fitted

import numpy as np

from ..core import array as ska
from ..metrics import prediction_quality, predict_reject_accuracy_score
from .base import RejectorMixin


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator."""

    def check(self):
        if hasattr(self, "estimator_"):
            # raise an AttributeError if `attr` does not exist
            getattr(self.estimator_, attr)
            return True
        # raise an AttributeError if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check


class BaseFallbackClassifier(
    BaseEstimator,
    MetaEstimatorMixin,
    RejectorMixin,
    metaclass=abc.ABCMeta,
):
    """An ABC for fallback meta-classifiers.

    Parameters
    ----------
    estimator : object
        The base estimator making decisions w/o fallbacks.
    fallback_label : any, default=-1
        The label of a rejected example.
        Should be compatible w/ the class labels from training data.
    fallback_mode : {"return", "store"}, default="store"
        While predicting, whether to return a numpy ndarray of both predictions and
        fallbacks, or an fbndarray of predictions storing also fallback mask.
    """

    _parameter_constraints = {
        "estimator": [HasMethods(["fit", "predict_proba"])],
    }

    _estimator_type = "rejector"

    def __init__(
        self,
        estimator,
        *,
        fallback_label=-1,
        fallback_mode="store",
    ):
        self.estimator = estimator
        self.fallback_label = fallback_label
        self.fallback_mode = fallback_mode

    def _set_fitted_attributes(self, fitted_params):
        """Sets meta-estimator attributes after successful train."""
        self.is_fitted_ = True

        for key, value in fitted_params.items():
            setattr(self, key, value)

    def _validate_fallback_label(self, classes_):
        """Checks if fallback label is compatible w/ classes from training data.."""
        fallback_label_ = self.fallback_label

        try:
            fallback_label_.dtype
        except AttributeError:
            fallback_label_ = np.asarray(
                fallback_label_,
                dtype=classes_.dtype,
            ).view(np.ndarray)

        if fallback_label_ in classes_:
            warnings.warn(
                (
                    f"Fallback label = {fallback_label_} is in fitted classes = "
                    f"{classes_}"
                ),
                category=UserWarning,
            )

        return fallback_label_

    def _fit(self, X, y, set_attributes=None, **fit_params):
        """Fits the base estimator."""
        estimator_ = clone(self.estimator)
        if "sample_weight" in fit_params:
            estimator_.fit(X, y, sample_weight=fit_params.get("sample_weight"))
        else:
            estimator_.fit(X, y)

        set_attributes = set_attributes or {}
        set_attributes.update({"estimator_": estimator_})
        return set_attributes

    @_fit_context(prefer_skip_nested_validation=False)
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "y": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X, y, **fit_params):
        """Trains base estimator and sets meta-estimator attributes.

        If ``X`` and ``y`` are valid, obtains unique classes and saves in
        ``self.classes_``. New ``self.fallback_label_`` is set to be of the same type
        as ``self.classes_``.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)

        classes_ = unique_labels(y)

        fallback_label_ = self._validate_fallback_label(classes_)

        set_attributes = {"fallback_label_": fallback_label_, "classes_": classes_}
        set_attributes = self._fit(X, y, set_attributes=set_attributes, **fit_params)
        self._set_fitted_attributes(set_attributes)

        return self

    @abc.abstractmethod
    def _predict(self, X):
        """An abstract method to make predictions w/ or w/o fallbacks."""

    @abc.abstractmethod
    def _set_fallback_mask(self, y_prob):
        """Sets the fallback mask for predicted probabilites."""

    @available_if(_estimator_has("predict"))
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(self, X):
        """Makes predictions using the base estimator and the fallback rules.

        Parameters
        ----------
        X : indexable, length n_samples)
            Input samples to classify.
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        Either an ndarray of predictions w/ optional fallback labels, or an FBNDarray of
        base-estimator predictions w/ fallback mask.
        """
        check_is_fitted(self, attributes="is_fitted_")
        return self._predict(X)

    @available_if(_estimator_has("predict_proba"))
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_proba(self, X):
        """Calls ``predict_proba`` on the estimator.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : FBNDArray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class probabilities for `X` based on the estimator.
            The order of the classes corresponds to that in the fitted
            attribute :term:`classes_`.
        """
        check_is_fitted(self, attributes="is_fitted_")
        y_prob = ska.fbarray(self.estimator_.predict_proba(X))
        self._set_fallback_mask(y_prob)
        return y_prob

    @available_if(_estimator_has("decision_function"))
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def decision_function(self, X):
        """Calls ``decision_function`` on the estimator.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,) or (n_samples, n_classes) \
                or (n_samples, n_classes * (n_classes-1) / 2)
            Result of the decision function for `X` based on the estimator.
        """
        check_is_fitted(self, attributes="is_fitted_")
        return self.estimator_.decision_function(X)

    @available_if(_estimator_has("predict_log_proba"))
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_log_proba(self, X):
        """Returns log of ``predict_proba`` on the estimator.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : FBNDArray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class log-probabilities for `X` based on the estimator.
            The order of the classes corresponds to that in the fitted
            attribute :term:`classes_`.
        """
        y_prob = self.predict_proba(X)
        return np.log(y_prob)

    @available_if(_estimator_has("score"))
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "y": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def score(self, X, y):
        """Returns the prediction-rejection accuracy score of the estimator.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to evaluate.
            Must fulfill the input assumptions of the
            underlying estimator.

        y : array-like of shape (n_samples,)
            True labels for `X` (excluding fallback label).

        Returns
        -------
        score : float
            Prediction-fallback accuracy.

        See also
        --------
        skfb.metrics.predict_reject_accuracy_score
        """
        y_pred = self.predict(X)
        if self.fallback_mode == "store":
            return predict_reject_accuracy_score(y, y_pred)
        return prediction_quality(y, y_pred, accuracy_score, self.fallback_label_)


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
        "fallback_mode": [StrOptions({"return", "store"})],
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
    fallback_mode : {"store", "return"}, default="return"
        If "store", returns an FBNDArray of shape (n_samples,) w/ estimator predictions
        and fallback mask attribute.
        If "return", returns an NDArray of shape (n_samples,) w/ estimator predictions
        and rejections.
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
    fallback_mode : {"return", "store"}, default="store"
        While predicting, whether to return a numpy ndarray of both predictions and
        fallbacks, or an fbndarray of predictions storing also fallback mask.

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

    def _set_fallback_mask(self, y_prob):
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


class ThresholdFallbackClassifierCV(ThresholdFallbackClassifier):
    """A fallback classifier based on the best certainty threshold learnt via CV.

    Parameters
    ----------
    estimator : object
        The base estimator making decisions.
    thresholds : array-like of shape (n_thresholds,), default=(0.1, 0.5, 0.9)
        Array of fallback thresholds to evaluate.
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
    scoring : callable, default=None
        A scorer callable object supporting a reject option, such as metrics from
        :mod:`~skfb.metrics`.
    verbose : int, default=0
        Verbosity level .
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
            "thresholds": ["array-like", Interval(Real, 0.0, 1.0, closed="both")],
            "cv": ["cv_object"],
            "scoring": [callable, None],
            "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
        },
    )

    def __init__(
        self,
        estimator,
        *,
        thresholds=(0.1, 0.5, 0.9),
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

        # region Validate and/or create objects for cv.
        cv_ = check_cv(self.cv, y=y, classifier=True)
        scoring_ = self.scoring or predict_reject_accuracy_score
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
            for threshold in self.thresholds
            for train_idx, test_idx in cv_.split(X, y)
        )
        cv_scores_ = np.array(scores).reshape(len(self.thresholds), cv_.n_splits)
        # endregion

        # region Update fitted attributes
        mean_cv_scores = cv_scores_.mean(axis=1)
        threshold_ = self.thresholds[np.argmax(mean_cv_scores)]
        best_score_ = mean_cv_scores.max()
        set_attributes.update(
            {
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

    def _set_fallback_mask(self, y_prob):
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
    fallback_rates : array-like of float, default=(0.1,)
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
        While predicting, whether to return a numpy ndarray of both predictions and
        fallbacks, or an fbndarray of predictions storing also fallback mask.

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
    ...     fallback_rates=(0.2,),
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
            "fallback_rates": ["array-like", Interval(Real, 0.0, 1.0, closed="both")],
            "cv": ["cv_object"],
            "scoring": [callable, None],
            "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
        },
    )

    def __init__(
        self,
        estimator,
        *,
        fallback_rates=(0.1,),
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

        self.fallback_rates = fallback_rates
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, X, y, set_attributes=None, **fit_params):
        """Learns the best threshold."""
        set_attributes = set_attributes or {}

        cv_ = check_cv(self.cv, y=y, classifier=True)
        set_attributes.update({"cv_": cv_})

        estimator = clone(self.estimator)

        thresholds_ = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, prefer="processes"
        )(
            delayed(_find_threshold)(
                estimator,
                X[train_idx],
                X[test_idx],
                y[train_idx],
                fallback_rate,
            )
            for fallback_rate in self.fallback_rates
            for train_idx, test_idx in cv_.split(X, y)
        )
        set_attributes.update(
            {
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

    def _set_fallback_mask(self, y_prob):
        """Sets the fallback mask for predicted probabilites."""
        y_prob.fallback_mask = y_prob.max(axis=1) < self.threshold_
