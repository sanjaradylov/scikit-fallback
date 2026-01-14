"""Supervised learning with ensembles and model router."""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_X_y

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
from ._common import fit_one, fit_and_predict_on_test
from ..core.array import earray


class SingleModelRouterClassifier(BaseEstimator, ClassifierMixin):
    """Defers input to the most appropriate classifier by selecting a single model per sample.

    Trains a pool of estimators and a router that learns to select the best
    estimator for each input based on a cost vector. The router is trained
    using cross-validated predictions from the estimators to determine which
    estimator is most appropriate for each input. If none of the estimators
    predict correctly, a default estimator by `default_index` is used. If multiple
    estimators predict correctly, the one with the lowest cost is chosen.

    Parameters
    ----------
    estimators : list of objects
        List of candidate estimators to choose from.
    router : object
        Classifier used to route inputs to estimators.
    costs : float or list of float, default=None
        List of costs associated with each estimator. If scalar, costs are uniform.
        If None, defaults to uniform 1.0.
    cv : int, cross-validation generator or an iterable, default=None
        Cross-validation strategy for training estimators and router.
    default_index : int, default=-1
        Index of the default estimator to use when no estimator predicts correctly.
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
        "default_index": [Integral],
        "return_earray": ["boolean"],
        "n_jobs": [Interval(Integral, -1, None, closed="left"), None],
    }

    def __init__(
        self,
        estimators,
        router,
        costs=None,
        cv=None,
        default_index=-1,
        return_earray=False,
        n_jobs=None,
    ):
        self.estimators = estimators
        self.router = router
        self.costs = costs
        self.cv = cv
        self.default_index = default_index
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
        - Validate inputs and determine classes.
        - Use cross-validated predictions from candidate estimators to
          build routing targets (best estimator index per sample).
        - Train the router on full data to predict chosen estimator index.
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
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        # region Check training data
        X, y = check_X_y(
            X,
            y,
            accept_sparse=True,
            allow_nd=True,
            dtype=None,
            ensure_2d=False,
        )

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        # endregion

        # Expose classes
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
        y_route = self.make_router_targets(X, y, sample_weight=sample_weight)
        # Fit router on inputs and router targets
        self.router_ = fit_one(self.router, X, y_route, sample_weight=sample_weight)

        # region Fit final estimators on full data
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_one)(estimator, X, y, sample_weight)
            for estimator in self.estimators
        )
        # endregion

        self.is_fitted_ = True
        return self

    def make_router_targets(
        self,
        X,
        y,
        sample_weight=None,
    ):
        """Builds data for router training.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Features to train both estimator pool and router.
        y : array-like, shape = n_samples
            Labels to train both estimator pool and router.
        sample_weight : array-like or None, default=None
        """
        n_samples, n_estimators = X.shape[0], len(self.estimators)
        # Predictions on folds of X by each estimator
        y_pred = np.empty((n_estimators, n_samples), dtype=y.dtype)

        # region Make predictions on X and save in y_pred
        for train_idx, test_idx in self.cv_.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            sw_train = None if sample_weight is None else sample_weight[train_idx]
            X_test = X[test_idx]

            # Parallelize fitting and prediction across estimators
            predictions = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_and_predict_on_test)(
                    estimator, X_train, y_train, sw_train, X_test
                )
                for estimator in self.estimators
            )

            # Store predictions in y_pred
            for i, preds in enumerate(predictions):
                y_pred[i, test_idx] = preds
        # endregion

        return self.collect_target_labels(y, y_pred)

    def collect_target_labels(self, y_true, Y_pred):
        """Collects routing target labels based on estimator predictions and costs."""
        n_estimators, n_samples = Y_pred.shape
        # Routing targets (default to self.default_index when every estimator is wrong)
        y_route = np.full(n_samples, self.default_index, dtype=int)
        # Correctness mask
        correct_mask = Y_pred == y_true[np.newaxis, :]
        # Getting the optimal estimator index based on costs and correctness mask
        utilities = np.full((n_estimators, n_samples), -np.inf, dtype=np.float64)
        utilities[correct_mask] = (correct_mask * (-self.costs_)[:, np.newaxis])[
            correct_mask
        ]
        # Did any estimator get a correct prediction?
        correct_any = correct_mask.any(axis=0)
        # If yes, assign the best estimator as a routing target
        y_route[correct_any] = utilities.argmax(axis=0)[correct_any]

        return y_route

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "y": ["array-like"],
            "sample_weight": ["array-like", None],
        },
        prefer_skip_nested_validation=True,
    )
    def score(self, X, y, sample_weight=None):
        """Score predictions using accuracy.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples to evaluate.
        y : array-like, shape (n_samples,)
            True labels for X.
        sample_weight : array-like, shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        score : float
            Accuracy score.
        """
        from sklearn.metrics import accuracy_score

        check_is_fitted(self, attributes="is_fitted_")
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

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
        y_score : ndarray
            Decision function values.
        """
        return self._predict(X, "decision_function")

    def _predict(self, X, method):
        check_is_fitted(self, attributes="is_fitted_")
        X = check_array(
            X,
            accept_sparse=True,
            allow_nd=True,
            dtype=None,
            ensure_2d=False,
        )

        y_route = self.router_.predict(X)

        n_samples, n_estimators = X.shape[0], len(self.estimators_)

        # Determine output shape by getting shape from first estimator.
        # This handles cases where decision_function may be 1D or 2D.
        sample = np.unique(y_route)[0]
        first_estimator = self.estimators_[sample]
        sample_X = X[y_route == sample][:1]
        sample_output = getattr(first_estimator, method)(sample_X)

        if method == "predict":
            y_pred = np.empty(n_samples, dtype=self.classes_.dtype)
        else:
            # For predict_proba, predict_log_proba, decision_function.
            output_shape = (
                (n_samples,) + sample_output.shape[1:]
                if sample_output.ndim > 1
                else (n_samples,)
            )
            y_pred = np.zeros(output_shape, dtype=np.float64)

        # Ensemble mask if `self.return_earray` is True.
        if self.return_earray:
            ensemble_mask = np.zeros((n_samples, n_estimators), dtype=np.bool_)

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
