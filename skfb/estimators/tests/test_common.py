"""Common utilities for testing."""


class TestFallbackEstimator:
    """Accepts and returns the same probability scores."""

    # pylint: disable=unused-argument
    def fit(self, X_unused=None, y_unused=None):
        self.is_fitted_ = True
        return self

    def predict_proba(self, y):
        return y
