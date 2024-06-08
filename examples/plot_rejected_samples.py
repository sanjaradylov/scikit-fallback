"""
==========================================================================
Learning to reject ambiguously labeled samples w/ RateFallbackClassifierCV
==========================================================================
"""

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skfb.estimators import RateFallbackClassifierCV

import matplotlib.pyplot as plt
import numpy as np


# region Generate data
n_in, n_out = 10_000, 5_000
# pylint: disable=unbalanced-tuple-unpacking
X_in, y_in = make_blobs(
    n_samples=n_in, cluster_std=0.5, centers=[(-3, 3), (3, -3)], random_state=0
)
# pylint: disable=unbalanced-tuple-unpacking
X_out, y_out = make_blobs(
    n_samples=n_out, cluster_std=0.7, centers=[(-0.2, 0.2), (0.2, -0.2)], random_state=0
)
X = np.concatenate([X_in, X_out])
y = np.concatenate([y_in, y_out])
in_mask = np.concatenate([np.array([False] * n_in), np.array([True] * n_out)])
X_train, X_test, y_train, y_test, in_mask_train, in_mask_test = train_test_split(
    X, y, in_mask, test_size=0.5, random_state=0, shuffle=True
)
# endregion

# region Train a rejector
estimator = LogisticRegression(C=10_000, random_state=0)
rejector = RateFallbackClassifierCV(
    estimator, fallback_rates=(1 / 4, 1 / 3, 1 / 2), cv=3
)
rejector.fit(X_train, y_train)
# endregion

# region Get combined predictions and fallback mask
y_score = rejector.predict_proba(X_test)
y_comb = rejector.predict(X_test)
accepted_mask = y_score.get_dense_neg_fallback_mask()
fallback_mask = ~accepted_mask
# endregion

# region Print and plot results
print(
    f"Rate of ambiguous samples (test): "
    f"{in_mask_test.sum() / len(in_mask_test) * 100:.1f}%"
)
print(f"Fallback rate (test): {y_score.fallback_rate * 100.0:.1f}%")
print(
    f"Accuracy on accepted samples (test): "
    f"{rejector.set_params(fallback_mode='return').score(X_test, y_test) * 100:.1f}%"
)
plt.scatter(
    X_test[accepted_mask, 0],
    X_test[accepted_mask, 1],
    c=y_comb[accepted_mask],
    alpha=0.85,
    edgecolor="k",
    cmap="Blues",
    label="Accepted samples",
)
plt.scatter(
    X_test[fallback_mask, 0],
    X_test[fallback_mask, 1],
    c=y_test[fallback_mask],
    alpha=0.6,
    marker="X",
    edgecolor="k",
    cmap="Blues",
    label="Rejected samples",
)
plt.legend()
plt.tight_layout()
plt.show()
# endregion
