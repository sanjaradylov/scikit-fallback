"""Classification based on custom functions (e.g., rule-based classification)."""

__all__ = ("FallbackRuleClassifier", "RuleClassifier")

import abc
import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from ..utils._legacy import _fit_context, validate_params
from .base import RejectorMixin


class RuleClassificationWarning(UserWarning):
    """Raised if validation of RuleClassifier and its subclasses causes errors."""


class RuleClassifier(BaseEstimator, ClassifierMixin, metaclass=abc.ABCMeta):
    """ABC that defines rule-based classification.

    Reimplement ``_predict`` by defining custom classification rules (e.g.,
    label a transaction fraudulent if a receiver is in a blacklist; otherwise,
    it's genuine). You can introduce new rule arguments either via reimplemented
    ``__init__`` or by passing them as ``kwargs`` during model instatiation.

    Parameters
    ----------
    validate : bool, default=False
        Whether to validate reimplemented prediction method. Validation means checking
        if prediction doesn't result in any exception.
    kwargs : dict, default=None
        Any (tunable) argument required to make predictions.
        Each argument is returned and can be set individually by ``get_params`` and
        ``set_params``, respectively.
    """

    _parameter_constraints = {"validate": [bool], "kwargs": [None, dict]}

    def __init__(self, *, validate=False, kwargs=None):
        self.validate = validate
        self.kwargs = kwargs

    @_fit_context(prefer_skip_nested_validation=False)
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "y": ["array-like", None],
            "classes": ["array-like", None],
            "sample_weight": ["array-like", None],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X, y=None, sample_weight=None):
        """Fits the estimator and sets fit attributes.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        if y is not None:
            self.classes_ = unique_labels(y)
        self._fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

    # pylint: disable=unused-argument
    def _fit(self, X, y, sample_weight=None):
        """Only fits the estimator.

        Should be reimplemented if a rule requires a learning mechanism.
        """
        if self.validate:
            self.validate_predict(X)
        return self

    @_fit_context(prefer_skip_nested_validation=False)
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "y": ["array-like", None],
            "classes": ["array-like", None],
            "sample_weight": ["array-like", None],
        },
        prefer_skip_nested_validation=True,
    )
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Fits the estimator partially."""
        if classes is not None:
            self.classes_ = classes
        self._partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

    # pylint: disable=unused-argument
    def _partial_fit(self, X, y=None, classes=None, sample_weight=None):
        """Only fits the estimator partially.

        Should be reimplemented if a rule requires a learning mechanism.
        """
        return self

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def validate_predict(self, X):
        """Validates inference methods.

        Raises
        ------
        An exception raised by one of the methods.
        """
        try:
            self.is_fitted_ = True
            self.predict(X)
        except Exception:
            warnings.warn(
                (
                    "Validation of {self.__class__.__name__}.predict resulted in"
                    " errors; please, check your implementations"
                ),
                category=RuleClassificationWarning,
            )
            delattr(self, "is_fitted_")
            raise

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(self, X):
        """Predicts hard labels.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted hard labels.
        """
        check_is_fitted(self, attributes="is_fitted_")

        if self.validate:
            check_is_fitted(self, attributes="is_fitted_")

        return np.asarray(self._predict(X))

    @abc.abstractmethod
    def _predict(self, X):
        """An abstract method to make rule-based predictions.

        Should be reimplemented. If applicable, use ``self.kwargs`` as
        additional input parameters to your rule.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to classify.
        """

    def get_params(self, deep=True):
        """Gets parameters for the estimator."""
        parameters = super().get_params(deep)
        parameters |= {**(self.kwargs or {})}
        return parameters

    def set_params(self, **params):
        """Sets the parameters of the estimator."""
        if self.kwargs:
            for key, value in params.items():
                if key in self.kwargs:
                    self.kwargs[key] = value
        return super().set_params(**params)


class FallbackRuleClassifier(RuleClassifier, RejectorMixin):
    """Rule-based fallback classification.

    Reimplement ``_predict`` by defining custom classification rules, including
    fallbacks (e.g., label a transaction fraudulent if a receiver is in a blacklist;
    otherwise, it's either genuine for regular transactions or anomalous for irregular
    ones). You can introduce new rule arguments either via reimplemented ``__init__``
    or by passing them as ``kwargs`` during model instatiation.

    Parameters
    ----------
    validate : bool, default=False
        Whether to validate reimplemented prediction method. Validation means checking
        if prediction doesn't result in any exception.
    fallback_label : any, default=-1
        Label returned by fallback rules.
    kwargs : dict, default=None
        Any (tunable) argument required to make predictions.
        Each argument is returned and can be set individually by ``get_params`` and
        ``set_params``, respectively.
    """

    def __init__(self, *, validate=False, fallback_label=-1, kwargs=None):
        super().__init__(validate=validate, kwargs=kwargs)

        self.fallback_label = fallback_label

    def _fit(self, X, y, sample_weight=None):
        if y is not None and hasattr(self, "classes_"):
            self.fallback_label_ = self.validate_fallback_label(
                self.fallback_label, self.classes_
            )
        return super()._fit(X, y, sample_weight)

    def _partial_fit(self, X, y=None, classes=None, sample_weight=None):
        if y is not None and classes is not None:
            self.fallback_label_ = self.validate_fallback_label(
                self.fallback_label, classes
            )
        return super()._partial_fit(X, y, classes, sample_weight)

    @_fit_context(prefer_skip_nested_validation=False)
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "y": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def score(self, X, y):
        """Evaluates an accuracy score on accepted samples.

        Parameters
        ----------
        X : indexable, length n_samples
            Input samples to evaluate.

        y : array-like of shape (n_samples,)
            True labels for `X` (excluding fallback label).

        Returns
        -------
        score : float
            Accuracy on samples not labeled as fallbacks.

        See also
        --------
        skfb.metrics.prediction_quality
        """
        # ??? Is it the right way to overcome circular imports?
        # pylint: disable=import-outside-toplevel
        from ..metrics._common import prediction_quality

        check_is_fitted(self, attributes="fallback_label_")

        y_pred = self.predict(X)

        return prediction_quality(y, y_pred, accuracy_score, self.fallback_label_)
