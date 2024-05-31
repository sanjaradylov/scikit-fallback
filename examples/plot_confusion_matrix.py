"""
=======================================================
Logistic regression w/ and w/o a reject option on MNIST
=======================================================

An example plot of :class:`skfb.metrics.PAConfusionMatrixDisplay`.
"""

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skfb.estimators import ThresholdFallbackClassifier
from skfb.metrics import PAConfusionMatrixDisplay


X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    shuffle=True,
    stratify=y,
)

estimator = LogisticRegression(
    C=0.01,
    penalty="l1",
    solver="saga",
    tol=0.01,
    random_state=0,
)
pipe = make_pipeline(StandardScaler(), estimator)
rejector = ThresholdFallbackClassifier(pipe, threshold=0.5, fallback_mode="return")
rejector.fit(X_train, y_train)

print("LogisticRegression without fallbacks:")
print(f"Test accuracy: {rejector.estimator_.score(X_train, y_train) * 100.0:.2f}%")
print(f"Test accuracy: {rejector.estimator_.score(X_test, y_test) * 100.0:.2f}%")

print("\nLogisticRegression with fallbacks:")
print(f"Test accuracy: {rejector.score(X_train, y_train) * 100.0:.2f}%")
print(f"Test accuracy: {rejector.score(X_test, y_test) * 100.0:.2f}%")

PAConfusionMatrixDisplay.from_estimator(rejector, X_test, y_test).plot()
plt.show()
