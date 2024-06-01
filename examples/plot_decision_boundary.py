"""
===========================================================================
Decision boundaries for LogisticrRegression and ThresholdFallbackClassifier
===========================================================================
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skfb.estimators import ThresholdFallbackClassifier


n, test_size = 4_000, 0.2
X, y = make_moons(n_samples=n, noise=0.4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=0,
    shuffle=True,
)

estimator = LogisticRegression(C=10_000, random_state=0)
rejector = ThresholdFallbackClassifier(estimator, threshold=0.75)
rejector.fit(X_train, y_train)

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), int(n * test_size)),
    np.linspace(X[:, 1].min(), X[:, 1].max(), int(n * test_size)),
)
grid = np.c_[xx.ravel(), yy.ravel()]

y_prob = rejector.predict_proba(grid)[:, 1].reshape(xx.shape)

cm, cm_bright = plt.cm.RdBu, ListedColormap(["#FF0000", "#0000FF"])
plt.contourf(xx, yy, y_prob, 100, cmap=cm)
plt.contour(
    xx,
    yy,
    y_prob,
    [rejector.threshold - 0.5, rejector.threshold],
    cmap=ListedColormap(["#00703c"]),
    linestyles="dashed",
)
plt.contour(xx, yy, y_prob, [0.5], cmap="Greys_r")
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    alpha=0.85,
    edgecolor="k",
    cmap=cm_bright,
)
plt.xlabel("$X_0$")
plt.ylabel("$X_1$")
plt.title("Decision boundaries of\nThresholdFallbackClassifier(LogisticRegression())")
plt.show()
