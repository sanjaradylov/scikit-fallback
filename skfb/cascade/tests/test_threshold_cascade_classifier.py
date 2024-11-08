"""Tests threshold-based cascade ensembles."""

import time

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from skfb.cascade import ThresholdCascadeClassifier


def test_threshold_cascade_classifier():
    """Tests ``TresholdCascadeClassifier``."""
    n_iterations = 1_000

    X = np.array(
        [[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2.0, 2.5], [2.0, 2.0], [2.5, 2.5]]
    )
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0])

    maxent = LogisticRegression(random_state=0)
    rf = RandomForestClassifier(random_state=0)
    cascade = ThresholdCascadeClassifier([maxent, rf], 0.8).fit(X, y)

    X = np.concat([X] * n_iterations)
    y = np.concat([y] * n_iterations)

    tic = time.perf_counter()
    assert cascade.score(X, y) == 1.0
    toc = time.perf_counter()
    cascade_time = toc - tic

    tic = time.perf_counter()
    assert cascade.set_estimators(0).score(X, y) == 0.75
    toc = time.perf_counter()
    maxent_time = toc - tic

    tic = time.perf_counter()
    assert cascade.set_estimators(1).score(X, y) == 1.0
    toc = time.perf_counter()
    rf_time = toc - tic

    assert rf_time > cascade_time > maxent_time
