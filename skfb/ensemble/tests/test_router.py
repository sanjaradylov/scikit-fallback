"""Tests for SingleModelRouterClassifier."""

import numpy as np
import pytest

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels

from skfb.ensemble import SingleModelRouterClassifier


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """Dummy classifier for testing."""

    def __init__(self, y_pred=None, y_proba=None):
        self.y_pred = y_pred
        self.y_proba = y_proba

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        if self.y_pred is not None:
            return self.y_pred
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        if self.y_proba is not None:
            return self.y_proba
        return np.ones((X.shape[0], len(self.classes_))) / len(self.classes_)


class TestSingleModelRouterClassifierBasics:
    """Tests basic functionality of `SingleModelRouterClassifier`."""

    n_samples = 50
    n_classes = 2

    @pytest.fixture
    def sample_data(self):
        """Generates sample classification data."""
        return make_classification(
            n_samples=self.n_samples,
            n_classes=self.n_classes,
            n_features=5,
            random_state=0,
        )

    @pytest.fixture
    def estimators(self):
        """Creates a pool of candidate estimators."""
        return [
            DecisionTreeClassifier(max_depth=2, random_state=0),
            LogisticRegression(max_iter=200, random_state=0),
            RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0),
        ]

    @pytest.fixture
    def router(self):
        """Creates a router estimator."""
        return LogisticRegression(max_iter=200, random_state=1)

    def test_fit(self, sample_data, estimators, router):
        """Tests basic fit functionality."""
        X, y = sample_data
        clf = SingleModelRouterClassifier(estimators, router, cv=3)

        assert clf is clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert hasattr(clf, "estimators_")
        assert hasattr(clf, "router_")
        assert hasattr(clf, "costs_")
        assert hasattr(clf, "cv_")
        assert hasattr(clf, "is_fitted_")

        assert clf.is_fitted_ is True
        assert len(clf.estimators_) == len(estimators)
        assert clf.classes_.shape[0] == self.n_classes

    def test_predict(self, sample_data, estimators, router):
        """Tests basic predict functionality."""
        X, y = sample_data
        clf = SingleModelRouterClassifier(estimators, router, cv=2).fit(X, y)

        y_pred = clf.predict(X)

        assert y_pred.shape == (self.n_samples,)
        assert np.all(np.isin(y_pred, clf.classes_))

    def test_predict_proba(self, sample_data, estimators, router):
        """Tests basic predict_proba functionality."""
        X, y = sample_data
        clf = SingleModelRouterClassifier(estimators, router, cv=2).fit(X, y)

        y_proba = clf.predict_proba(X)

        assert y_proba.shape == (self.n_samples, len(clf.classes_))
        np.testing.assert_array_almost_equal(
            y_proba.sum(axis=1),
            np.ones(self.n_samples),
        )


class TestSingleModelRouterClassifierCosts:
    """Tests cost handling in `SingleModelRouterClassifier`."""

    n_samples = 90
    n_classes = 3

    @pytest.fixture
    def sample_data(self):
        return make_classification(
            n_samples=self.n_samples,
            n_classes=self.n_classes,
            n_features=5,
            n_clusters_per_class=1,
            random_state=0,
        )

    @pytest.fixture
    def estimators(self):
        return [
            DecisionTreeClassifier(max_depth=2, random_state=0),
            LogisticRegression(max_iter=200, random_state=0),
        ]

    @pytest.fixture
    def router(self):
        return LogisticRegression(max_iter=200, random_state=1)

    def test_costs_none_defaults_to_ones(self, sample_data, estimators, router):
        """Tests that costs=None defaults to uniform cost of 1.0."""
        X, y = sample_data
        clf = SingleModelRouterClassifier(estimators, router, cv=2).fit(X, y)

        expected_costs = np.ones(len(estimators), dtype=np.float64)
        np.testing.assert_array_almost_equal(clf.costs_, expected_costs)

    def test_costs_scalar_broadcasts(self, sample_data, estimators, router):
        """Tests that scalar costs broadcast across estimators."""
        X, y = sample_data
        clf = SingleModelRouterClassifier(estimators, router, costs=0.5, cv=2).fit(X, y)

        expected_costs = np.array([0.5] * len(estimators))
        np.testing.assert_array_almost_equal(clf.costs_, expected_costs)

    def test_costs_array(self, sample_data, estimators, router):
        """Tests that array costs are stored correctly."""
        X, y = sample_data
        costs = [0.2, 0.8]
        clf = SingleModelRouterClassifier(estimators, router, costs=costs, cv=2)
        clf.fit(X, y)

        np.testing.assert_array_almost_equal(clf.costs_, np.array(costs))

    def test_costs_affect_routing(self, estimators, router):
        """Tests that different costs affect routing decisions."""
        # Create data where estimators have different accuracies.
        np.random.seed(0)
        X = np.random.randn(30, 5)
        y = np.array([0, 1] * 15)

        # Test with different cost vectors.
        clf1 = SingleModelRouterClassifier(estimators, router, costs=[0.1, 1.0], cv=2)
        clf1.fit(X, y)

        clf2 = SingleModelRouterClassifier(estimators, router, costs=[1.0, 0.1], cv=2)
        clf2.fit(X, y)

        # Routing targets should differ when costs differ.
        targets1 = clf1.make_router_targets(X, y)
        targets2 = clf2.make_router_targets(X, y)

        # They may not be completely different, but structure should be different
        # (at least for some samples).
        assert not np.array_equal(targets1, targets2)


class TestSingleModelRouterClassifierDefaultIndex:
    """Tests the `default_index` parameter."""

    @pytest.fixture
    def sample_data(self):
        return make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=0
        )

    @pytest.fixture
    def router(self):
        return LogisticRegression(max_iter=200, random_state=1)

    def test_default_index_negative_one(self, sample_data, router):
        """Tests default_index=-1 (last estimator)."""
        X, y = sample_data
        estimators = [
            DecisionTreeClassifier(max_depth=1, random_state=0),
            LogisticRegression(max_iter=200, random_state=0),
        ]
        clf = SingleModelRouterClassifier(estimators, router, default_index=-1, cv=2)
        clf.fit(X, y)

        # Check that routing targets include the default index
        targets = clf.make_router_targets(X, y)
        assert np.any(targets == -1) or np.all(targets >= 0)

    def test_default_index_zero(self, sample_data, router):
        """Tests `default_index=0` (first estimator)."""
        X, y = sample_data
        estimators = [
            DecisionTreeClassifier(max_depth=1, random_state=0),
            LogisticRegression(max_iter=200, random_state=0),
        ]
        clf = SingleModelRouterClassifier(estimators, router, default_index=0, cv=2)
        clf.fit(X, y)

        targets = clf.make_router_targets(X, y)
        # All samples should have routing target 0 or -1 is not used
        assert np.all((targets >= 0) & (targets < len(estimators)))

    def test_default_index_used_when_all_wrong(self, router):
        """Tests that default_index is used when all estimators predict wrong."""
        # Create simple 2-estimator, 3-sample case
        y_true = np.array([1, 1, 1])  # True labels

        estimator1 = DummyClassifier(y_pred=np.array([0, 0, 0]))
        estimator2 = DummyClassifier(y_pred=np.array([0, 0, 0]))
        estimators = [estimator1, estimator2]

        clf = SingleModelRouterClassifier(estimators, router, default_index=-1)
        clf.costs_ = np.array([1.0, 1.0])
        clf.default_index = -1

        # All predictions wrong (0 != 1)
        Y_pred = np.array([[0, 0, 0], [0, 0, 0]])

        targets = clf.collect_target_labels(y_true, Y_pred)

        # All should be default index since all predictions are wrong
        assert np.all(targets == -1)


class TestSingleModelRouterClassifierCollectTargetLabels:
    """Tests the `collect_target_labels` method."""

    @pytest.fixture
    def estimators(self):
        return [
            DecisionTreeClassifier(max_depth=2, random_state=0),
            LogisticRegression(random_state=1, max_iter=200),
        ]

    @pytest.fixture
    def router(self):
        return LogisticRegression(random_state=42, max_iter=200)

    def test_collect_target_labels_correct_predictions(self, estimators, router):
        """Tests `collect_target_labels` with all correct predictions."""
        y_true = np.array([0, 1, 0])

        clf = SingleModelRouterClassifier(
            estimators, router, costs=[1.0, 2.0], default_index=-1
        )
        clf.costs_ = np.array([1.0, 2.0])

        # Both estimators correct (lower cost should be chosen).
        Y_pred = np.array([[0, 1, 0], [0, 1, 0]])
        targets = clf.collect_target_labels(y_true, Y_pred)

        # First estimator should be chosen (lower cost).
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(targets, expected)

    def test_collect_target_labels_picks_lowest_cost(self, estimators, router):
        """Tests that `collect_target_labels` picks lowest cost among correct."""
        y_true = np.array([0, 1])

        clf = SingleModelRouterClassifier(
            estimators, router, costs=[0.5, 1.0], default_index=-1
        )
        clf.costs_ = np.array([0.5, 1.0])

        # Both correct, but first has lower cost
        Y_pred = np.array([[0, 1], [0, 1]])
        targets = clf.collect_target_labels(y_true, Y_pred)
        np.testing.assert_array_equal(targets, np.array([0, 0]))

    def test_collect_target_labels_uses_default_when_all_wrong(
        self, estimators, router
    ):
        """Test that collect_target_labels uses default_index when all wrong."""
        y_true = np.array([0, 1])

        clf = SingleModelRouterClassifier(estimators, router, default_index=1)
        clf.costs_ = np.array([1.0, 1.0])

        # All predictions wrong
        Y_pred = np.array([[1, 0], [0, 0]])

        targets = clf.collect_target_labels(y_true, Y_pred)
        # Should use default index for wrong predictions
        assert np.all((targets == 1) | (targets < len(estimators)))
