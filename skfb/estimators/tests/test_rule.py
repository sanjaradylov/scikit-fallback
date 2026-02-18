"""Tests rule-based classifiers."""

import numpy as np

from skfb.estimators import FallbackRuleClassifier, RuleClassifier


class AgeVsDefaultClassifier(RuleClassifier):
    """Predicts 0/1 based on min and max values of feature."""

    def __init__(self, validate=False, min_threshold=19, max_threshold=70, kwargs=None):
        super().__init__(validate=validate, kwargs=kwargs)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def _predict(self, X):
        y = np.zeros(X.shape[0])
        y[(X[:, 0] < self.min_threshold) | (X[:, 0] > self.max_threshold)] = 1
        return y


def test_custom_rule_classifier():
    X = np.array([[28, -1], [20, -10], [78, 4], [56, 0], [33, -12], [74, 44], [20, 0]])
    y = np.array([0, 1, 0, 0, 0, 1, 1])
    # y_pred =   [0, 0, 1, 0, 0, 1, 0]
    clf = AgeVsDefaultClassifier(validate=True).fit(X)

    np.testing.assert_array_equal(clf.predict(X), [0, 0, 1, 0, 0, 1, 0])
    assert clf.score(X, y) == 4 / 7
    assert clf.get_params() == {
        "validate": True,
        "min_threshold": 19,
        "max_threshold": 70,
        "kwargs": None,
    }
    assert clf.set_params(min_threshold=21).score(X, y) == 6 / 7


class AgeVsDefaultFallbackClassifier(FallbackRuleClassifier):
    """Similar to AgeVsDefaultClassifier but also predicts fallbacks."""

    def __init__(
        self,
        validate=False,
        fallback_label=-1,
        min_threshold=19,
        max_threshold=70,
        kwargs=None,
    ):
        super().__init__(
            validate=validate, fallback_label=fallback_label, kwargs=kwargs
        )
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def _predict(self, X):
        y = np.zeros(X.shape[0])
        y[(X[:, 0] < self.min_threshold) | (X[:, 0] > self.max_threshold)] = 1
        y[np.isnan(X[:, 1])] = -1
        return y


def test_custom_fallback_rule_classifier():
    X = np.array(
        [[28, np.nan], [20, -10], [78, 4], [56, 0], [33, -12], [74, 44], [20, np.nan]]
    )
    y = np.array([0, 1, 0, 0, 0, 1, 1])
    clf = AgeVsDefaultFallbackClassifier(validate=True).fit(X, y)

    np.testing.assert_array_equal(clf.predict(X), [-1, 0, 1, 0, 0, 1, -1])
    assert clf.score(X, y) == 3 / 5
