"""Supervised learning with ensembles and model router."""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, check_array, check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_array

try:
    from sklearn.utils.parallel import delayed, Parallel
except ModuleNotFoundError:
    from joblib import Parallel

    # pylint: disable=ungrouped-imports
    from sklearn.utils.fixes import delayed

from ..utils._legacy import (
    _fit_context,
    HasMethods,
    Integral,
    Interval,
    Real,
    validate_params,
)
from ._common import fit_one, fit_and_predict_one_on_test
from ..core.array import earray


class RoutingClassifier(BaseEstimator, ClassifierMixin):
    """Defers input to the most appropriate classifier chosen through semantic routing.

    Trains a pool of `estimators` and a `router` that learns to select the best
    estimator for each input based on a `costs` vector. The router is trained
    using cross-validated predictions from the estimators to determine which
    estimator is most appropriate for each input.

    Parameters
    ----------
    estimators : list of objects
        List of candidate estimators to choose from.
    router : object
        Classifier used to route inputs to estimators.
    costs : float or list of float, default=None
        List of costs associated with each estimator (positive, higher is more costly).
        If scalar, costs are uniform. If None, defaults to uniform 1.0.
    cv : int, cross-validation generator or an iterable, default=None
        Cross-validation strategy for training estimators and router.
    return_earray : bool, default=False
        Whether to return :class:`~skfb.core.array.ENDArray` of predicted classes
        or plain numpy ndarray. ENDArray tracks which estimator made each prediction.
    n_jobs : int, default=None
        Number of jobs to run in parallel for cross-validation.
        If None, use 1.
    """

    _parameter_constraints = {
        "estimators": ["array-like"],
        "router": [HasMethods(["fit", "predict"])],
        "costs": ["array-like", Interval(Real, 0, None, closed="left"), None],
        "cv": ["cv_object", None],
        "return_earray": ["boolean"],
        "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
    }

    def __init__(
        self,
        estimators,
        router,
        costs=None,
        cv=None,
        return_earray=False,
        n_jobs=None,
    ):
        self.estimators = estimators
        self.router = router
        self.costs = costs
        self.cv = cv
        self.return_earray = return_earray
        self.n_jobs = n_jobs

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
        """Trains estimators and router.

        Steps:
        - Use cross-validated predictions from candidate estimators to
          build routing targets (best estimator index per sample).
        - Train the router on full data and store it in `self.router_`
          to predict chosen estimator index.
        - Fit all candidate estimators on full data and store them in
          `self.estimators_` for inference.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        sample_weight : array-like, shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_ = unique_labels(y)

        # region Normalize costs
        if self.costs is None:
            costs_ = np.ones(len(self.estimators), dtype=np.float64)
        elif isinstance(self.costs, float):
            costs_ = [self.costs] * len(self.estimators)
        else:
            costs_ = self.costs
        self.costs_ = np.asarray(costs_, dtype=np.float64)
        # endregion

        # Validate and/or create cv.
        self.cv_ = check_cv(self.cv, y=y, classifier=True)

        # Build router targets using cross-validated predictions
        y_route = self._make_router_targets(X, y, sample_weight=sample_weight)
        # Fit router on inputs and router targets
        self.router_ = fit_one(self.router, X, y_route, sample_weight=sample_weight)

        # region Fit final estimators on full data
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_one)(estimator, X, y, sample_weight)
            for estimator in self.estimators
        )
        # endregion

        self.is_fitted_ = True
        self._route_target_type = type_of_target(y_route)

        return self

    def _make_router_targets(
        self,
        X,
        y,
        sample_weight=None,
    ):
        """Builds data for router training."""
        n_samples = len(X)
        n_estimators = len(self.estimators)
        n_classes = len(np.unique(y))

        # Accumulate probabilities (handle repeated CV folds)
        y_proba_sum = np.zeros((n_estimators, n_samples, n_classes))
        fold_counts = np.zeros(n_samples, dtype=int)

        for train_idx, test_idx in self.cv_.split(X, y):
            X_train = np.take(X, train_idx, axis=0)
            y_train = np.take(y, train_idx, axis=0)
            X_test = np.take(X, test_idx, axis=0)
            if sample_weight is not None:
                sw_train = np.take(sample_weight, train_idx, axis=0)
            else:
                sw_train = None

            # Get probability predictions from each estimator
            probas = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_and_predict_one_on_test)(
                    estimator, X_train, y_train, sw_train, X_test, "predict_proba",
                )
                for estimator in self.estimators
            )

            # Accumulate for averaging
            for i, proba in enumerate(probas):
                y_proba_sum[i, test_idx, :] += proba
            fold_counts[test_idx] += 1

        # Average probabilities across folds
        y_proba = y_proba_sum / fold_counts[np.newaxis, :, np.newaxis]

        return self._collect_target_labels(y, y_proba)

    def _collect_target_labels(self, y_true, y_proba):
        """Collects routing target labels based on estimator predictions and costs."""
        n_estimators, n_samples = y_proba.shape[0], y_proba.shape[1]
        sample_idx = np.arange(n_samples)

        # Compute log-loss: -log(P(true_class))
        log_losses = np.zeros((n_estimators, n_samples))
        for i in range(n_estimators):
            true_class_proba = y_proba[i, sample_idx, y_true]
            # Clip to avoid log(0) / log(epsilon)
            true_class_proba = np.clip(true_class_proba, 1e-15, 1.0 - 1e-15)
            log_losses[i] = -np.log(true_class_proba)

        # Scale losses by cost and find best estimator per sample
        cost_scaled_losses = log_losses * self.costs_[:, np.newaxis]
        y_route = cost_scaled_losses.argmin(axis=0)

        router_classes, router_class_counts = np.unique(y_route, return_counts=True)
        self.router_class_ratios_ = dict(
            zip(router_classes, router_class_counts / sum(router_class_counts))
        )

        return y_route

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(self, X):
        """Predicts class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        return self._predict(X, "predict")

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_proba(self, X):
        """Predicts class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        y_proba : ndarray, shape (n_samples, n_classes)
            Class probabilities.
        """
        return self._predict(X, "predict_proba")

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_log_proba(self, X):
        """Predicts log class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        y_log_proba : ndarray, shape (n_samples, n_classes)
            Log class probabilities.
        """
        return np.log(self.predict_proba(X))

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
        },
        prefer_skip_nested_validation=True,
    )
    def decision_function(self, X):
        """Compute decision function for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        y_score : ndarray, shape (n_samples, n_classes) or (n_samples,)
            Decision function values.
        """
        return self._predict(X, "decision_function")

    def _predict(self, X, method):
        """Main function to route inputs and make predictions."""
        check_is_fitted(self, attributes="is_fitted_")
        # Ensure we can extract input metadata
        try:
            X = check_array(
                X,
                accept_sparse=True,
                dtype=None,
                ensure_2d=False,
                ensure_all_finite=False,
                allow_nd=True,
            )
        except TypeError:
            X = check_array(
                X,
                accept_sparse=True,
                dtype=None,
                ensure_2d=False,
                force_all_finite=False,
                allow_nd=True,
            )

        # Choose estimators
        y_route = self.router_.predict(X)

        # Create prediction matrix
        n_samples, n_estimators = len(X), len(self.estimators_)

        if method == "predict":
            y_pred = np.empty(n_samples, dtype=self.classes_.dtype)
        else:
            if self._route_target_type == "binary" and method == "decision_function":
                output_shape = (n_samples,)
            else:
                output_shape = (n_samples, len(self.classes_))
            y_pred = np.zeros(output_shape, dtype=np.float64)

        # Ensemble mask if `self.return_earray` is True.
        if self.return_earray:
            ensemble_mask = np.zeros((n_samples, n_estimators), dtype=np.bool_)

        # Predict classes for each chosen estimator
        for estimator_idx in np.unique(y_route):
            estimator_mask = y_route == estimator_idx
            if not np.any(estimator_mask):
                continue

            estimator = self.estimators_[estimator_idx]
            y_chunk = getattr(estimator, method)(X[estimator_mask])
            y_pred[estimator_mask] = y_chunk

            if self.return_earray:
                ensemble_mask[estimator_mask, estimator_idx] = True

        if self.return_earray:
            return earray(y_pred, ensemble_mask)
        else:
            return y_pred
