"""Base classes for estimators w/ a rejection option."""

__all__ = (
    "is_rejector",
    "BaseFallbackClassifier",
    "RejectorMixin",
)

import abc
import warnings

from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted

import numpy as np

from ..core import array as ska
from ..utils._legacy import (
    _fit_context,
    HasMethods,
    StrOptions,
    validate_params,
)


class RejectorMixin:
    """Mixin class for estimators w/ a rejection option."""

    _estimator_type = "rejector"

    def _more_tags(self):
        """For now, rejection-based learning is supervised."""
        return {"requires_y": True}


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
    fallback_mode : {"return", "store", "ignore"}, default="store"
        While predicting, whether to return:

        * (``"return"``) a numpy ndarray of both predictions and fallbacks;
        * (``"store"``)  an FBNDArray of predictions storing also fallback mask;
        * (``"ignore"``) a numpy ndarray of only estimator's predictions.
    """

    _parameter_constraints = {
        "estimator": [HasMethods(["fit", "predict_proba"])],
        "fallback_mode": [StrOptions({"return", "store", "ignore"})],
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
    def _set_fallback_mask(self, y_prob, X=None):
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
        y_pred : ndarray or FBNDArray of shape (n_samples,)
            Depending on ``self.fallback_mode``:

            * (``"return"``) a numpy ndarray of both predictions and fallbacks, or;
            * (``"store"``)  an FBNDArray of predictions storing also fallback mask, or;
            * (``"ignore"``) a numpy ndarray of only estimator's predictions.
        """
        check_is_fitted(self, attributes="is_fitted_")

        if self.fallback_mode == "ignore":
            return self.estimator_.predict(X)
        else:
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

        If fallback_mode != "ignore", returns FBNDArray w/ fallback mask.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : FBNDArray or ndarray of shape (n_samples, n_classes)
            Predicted class probabilities for `X` based on the estimator.
            The order of the classes corresponds to that in the fitted
            attribute :term:`classes_`.
            If ``self.fallback_mode == "ignore"``, returns an ndarray.
        """
        check_is_fitted(self, attributes="is_fitted_")

        if self.fallback_mode == "ignore":
            y_prob = self.estimator_.predict_proba(X)
        else:
            y_prob = ska.fbarray(self.estimator_.predict_proba(X))
            self._set_fallback_mask(y_prob, X=X)

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

        If fallback_mode != "ignore", returns FBNDArray w/ fallback mask.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : FBNDArray or ndarray of shape (n_samples, n_classes)
            Predicted class log-probabilities for `X` based on the estimator.
            The order of the classes corresponds to that in the fitted
            attribute :term:`classes_`.
            If ``self.fallback_mode == "ignore"``, returns an ndarray.
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
        """Evaluates an accuracy score.

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
            Depending on ``self.fallback_mode``:

            * (``"return"``) a numpy ndarray of both predictions and fallbacks;
            * (``"store"``)  an FBNDArray of predictions storing also fallback mask;
            * (``"ignore"``) a numpy ndarray of only estimator's predictions.

        See also
        --------
        skfb.metrics.predict_reject_accuracy_score
        """
        # ??? Is it the right way to overcome circular imports?
        # pylint: disable=import-outside-toplevel
        from ..metrics._classification import predict_reject_accuracy_score
        from ..metrics._common import prediction_quality

        y_pred = self.predict(X)

        if self.fallback_mode == "store":
            return predict_reject_accuracy_score(y, y_pred)
        elif self.fallback_mode == "return":
            return prediction_quality(y, y_pred, accuracy_score, self.fallback_label_)
        else:
            return accuracy_score(y, y_pred)


def is_rejector(estimator):
    """Returns True if ``estimator`` provides a rejection option."""
    return getattr(estimator, "_estimator_type") == "rejector"
