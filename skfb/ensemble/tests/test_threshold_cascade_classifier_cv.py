"""Tests for ThresholdCascadeClassifierCV."""

import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skfb.core.exceptions import SKFBException, SKFBWarning
from skfb.ensemble import ThresholdCascadeClassifierCV


class TestThresholdCascadeClassifierCV:
    """Tests basic functionality of ThresholdCascadeClassifierCV."""

    @pytest.fixture
    def sample_data(self):
        """Generates sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            n_classes=2,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def estimators(self):
        """Creates three estimators with varying complexity."""
        return [
            LogisticRegression(random_state=42, max_iter=100),
            DecisionTreeClassifier(random_state=42),
            RandomForestClassifier(n_estimators=5, random_state=42),
        ]

    @pytest.fixture
    def costs(self):
        """Computational costs for estimators."""
        return np.array([1.0, 5.0, 10.0])

    def test_fit_basic(self, sample_data, estimators, costs):
        """Test basic fit functionality."""
        X, y = sample_data

        cascade = ThresholdCascadeClassifierCV(
            estimators=estimators,
            costs=costs,
            cv=3,
            cv_thresholds=3,
        ).fit(X, y)

        assert hasattr(cascade, "estimators_")
        assert hasattr(cascade, "costs_")
        assert hasattr(cascade, "cv_")
        assert hasattr(cascade, "scoring_")
        assert hasattr(cascade, "thresholds_")
        assert hasattr(cascade, "best_thresholds_")
        assert hasattr(cascade, "all_cv_thresholds_")
        assert hasattr(cascade, "mean_cv_scores_")
        assert hasattr(cascade, "mean_cv_costs_")
        assert cascade.is_fitted_

    def test_predict_basic(self, sample_data, estimators, costs):
        """Test basic predict functionality."""
        X, y = sample_data
        cascade = ThresholdCascadeClassifierCV(
            estimators=estimators,
            costs=costs,
            cv=3,
            cv_thresholds=3,
        )
        cascade.fit(X, y)
        y_pred = cascade.predict(X[:10])

        assert y_pred.shape == (10,)
        assert np.all(np.isin(y_pred, cascade.classes_))

    def test_pareto_front_computed(self, sample_data, estimators, costs):
        """Test that Pareto front is computed and non-empty."""
        X, y = sample_data
        cascade = ThresholdCascadeClassifierCV(
            estimators=estimators,
            costs=costs,
            cv=3,
            cv_thresholds=3,
        )
        cascade.fit(X, y)

        assert len(cascade.best_thresholds_) == len(cascade.estimators_) - 1
        assert len(cascade.thresholds_) == len(cascade.estimators_)
        assert len(cascade.all_cv_thresholds_) == cascade.cv_.n_splits**2
        assert len(cascade.mean_cv_scores_) == len(cascade.all_cv_thresholds_)
        assert len(cascade.mean_cv_costs_) == len(cascade.all_cv_thresholds_)

    def test_min_score_constraint_satisfied(self, sample_data, estimators, costs):
        """Test that min_score constraint selects feasible config."""
        X, y = sample_data
        min_score = 0.5

        cascade = ThresholdCascadeClassifierCV(
            estimators=estimators,
            costs=costs,
            cv=3,
            cv_thresholds=3,
            min_score=min_score,
        ).fit(X, y)

        best_thresholds = [float(t) for t in cascade.best_thresholds_]
        idx = cascade.all_cv_thresholds_.tolist().index(best_thresholds)
        assert cascade.mean_cv_scores_[idx] >= min_score - 1e-6

    def test_max_cost_constraint_satisfied(self, sample_data, estimators, costs):
        """Test that max_cost constraint selects feasible config."""
        X, y = sample_data
        max_cost = 20.0  # Allow all estimators

        cascade = ThresholdCascadeClassifierCV(
            estimators=estimators,
            costs=costs,
            cv=3,
            cv_thresholds=3,
            max_cost=max_cost,
        ).fit(X, y)

        best_thresholds = [float(t) for t in cascade.best_thresholds_]
        idx = cascade.all_cv_thresholds_.tolist().index(best_thresholds)
        assert cascade.mean_cv_costs_[idx] <= max_cost + 1e-6

    def test_constraint_violation_raises(self, sample_data, estimators, costs):
        """Test that constraint violation raises with raise_error=True."""
        X, y = sample_data

        cascade = ThresholdCascadeClassifierCV(
            estimators=estimators,
            costs=costs,
            cv=3,
            cv_thresholds=3,
            max_cost=0.5,  # Impossible: all estimators cost more
            raise_error=True,
        )

        with pytest.raises(SKFBException):
            cascade.fit(X, y)

    def test_constraint_violation_warns(self, sample_data, estimators, costs):
        """Test that constraint violation warns with raise_error=False."""
        X, y = sample_data
        max_cost = 0.5  # Impossible constraint

        cascade = ThresholdCascadeClassifierCV(
            estimators=estimators,
            costs=costs,
            cv=3,
            cv_thresholds=3,
            max_cost=max_cost,
            raise_error=False,
        )

        with pytest.warns(SKFBWarning):
            cascade.fit(X, y)

        assert cascade.is_fitted_

    def test_dual_constraints(self, sample_data, estimators, costs):
        """Test fitting with both min_score and max_cost constraints."""
        X, y = sample_data
        min_score, max_cost = 0.5, 20.0

        cascade = ThresholdCascadeClassifierCV(
            estimators=estimators,
            costs=costs,
            cv=3,
            cv_thresholds=3,
            min_score=min_score,
            max_cost=max_cost,
        ).fit(X, y)

        best_thresholds = [float(t) for t in cascade.best_thresholds_]
        idx = cascade.all_cv_thresholds_.tolist().index(best_thresholds)
        assert cascade.mean_cv_scores_[idx] >= min_score - 1e-6
        assert cascade.mean_cv_costs_[idx] <= max_cost + 1e-6

    def test_set_params_updates_constraints(self, sample_data, estimators, costs):
        """Test that set_params updates constraint parameters."""
        X, y = sample_data

        cascade = ThresholdCascadeClassifierCV(
            estimators=estimators,
            costs=costs,
            cv=3,
            cv_thresholds=3,
        ).fit(X, y)

        cascade.set_params(min_score=0.75, max_cost=10.0)
        assert cascade.min_score == 0.75
        assert cascade.max_cost == 10.0
