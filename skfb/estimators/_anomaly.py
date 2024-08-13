"""Fallback classification via anomaly detection."""

__all__ = ("AnomalyFallbackClassifier",)

import numpy as np

# pylint: disable=import-error,no-name-in-module
# pyright: reportMissingModuleSource=false
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ..core import array as ska
from ..utils._legacy import validate_params
from .base import BaseFallbackClassifier, _estimator_has


class AnomalyFallbackClassifier(BaseFallbackClassifier):
    """A fallback classifier based on provided anomaly detector.

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.estimators import AnomalyFallbackClassifier
    >>> from sklearn.ensemble import IsolationForest
    >>> from sklearn.linear_model import LogisticRegression
    >>> estimator = LogisticRegression(random_state=0)
    >>> outlier_detector = IsolationForest(n_estimators=10, max_samples=1.0,
    ...                                    contamination=0.2, random_state=0)
    >>> rejector = AnomalyFallbackClassifier(estimator, outlier_detector)
    >>> X = np.array([
    ...     [0, 0], [10, 10], [1, 1], [9, 9], [1, 0], [9, 10], [0, 1], [10, 9],
    ...     [5.5, 5], [5., 5.5]
    ... ])
    >>> y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    >>> rejector.fit(X, y).predict(X).get_dense_fallback_mask()
    array([False, False, False, False, False, False, False, False,  True,
            True])
    >>> rejector.set_params(fallback_mode="return").predict(X)
    array([ 0,  1,  0,  1,  0,  1,  0,  1, -1, -1])
    >>> rejector.score(X, y)
    1.0
    """

    _parameter_constraints = {**BaseFallbackClassifier._parameter_constraints}
    _parameter_constraints.update(
        {
            "outlier_detector": [HasMethods(["fit_predict"])],
        },
    )

    def __init__(
        self,
        estimator,
        outlier_detector,
        fallback_label=-1,
        fallback_mode="store",
    ):
        super().__init__(
            estimator=estimator,
            fallback_label=fallback_label,
            fallback_mode=fallback_mode,
        )

        self.outlier_detector = outlier_detector

    def fit(self, X, y, **fit_params):
        """Trains base estimator and outlier detector then sets fit attributes.

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
        self.outlier_detector.fit(X, y, **fit_params)
        self._set_fitted_attributes({"outlier_detector_": self.outlier_detector})
        return super().fit(X, y, **fit_params)

    def _predict(self, X):
        """Runs outlier detection and classification.

        Returns both fallbacks and classes if ``self.fallback_mask == 'return'``,
        or classes w/ fallback mask if ``self.fallback_mask == 'store'``.
        """
        y_out = self.outlier_detector_.predict(X)
        fallback_mask = y_out == -1

        if self.fallback_mode == "return":
            y_comb = np.empty(len(X), dtype=self.classes_.dtype)
            acceptance_mask = ~fallback_mask
            y_comb[acceptance_mask] = self.estimator_.predict(X[acceptance_mask])
            y_comb[fallback_mask] = self.fallback_label_
            return y_comb
        else:
            y_pred = self.estimator_.predict(X)
            y_pred = ska.fbarray(y_pred, fallback_mask)
            return y_pred

    def _set_fallback_mask(self, y_prob, X):
        """Doesn't set fallback mask for ``predict_proba`` and ``decision_function``."""
        y_out = self.outlier_detector_.predict(X)
        y_prob.fallback_mask = y_out == -1

    @available_if(_estimator_has("decision_function"))
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def decision_function(self, X):
        """Calls ``decision_function`` on the estimator and sets fallback mask.

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
        y_prob = ska.fbarray(self.estimator_.decision_function(X))
        self._set_fallback_mask(y_prob, X=X)
        return y_prob