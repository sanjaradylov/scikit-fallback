"""
=======================================
MLP w/ and w/o a reject option on MNIST
=======================================

An example plot of :class:`skfb.metrics.PAConfusionMatrixDisplay`.
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skfb.estimators import ThresholdFallbackClassifier
from skfb.metrics import PAConfusionMatrixDisplay


# pylint: disable=redefined-outer-name
def generate_random_combination(X):
    """Generates random combination of two images from different class."""
    alpha = np.random.uniform(0.2, 0.8)
    first = X[np.random.choice(X.shape[0])]
    second = X[np.random.choice(X.shape[0])]
    random = alpha * first + (1 - alpha) * second

    add_noise = np.random.choice(2)
    if add_noise:
        random = np.random.randint(0, 256, size=random.shape) + random
        random = np.clip(random, 0, 256)

    return random


# region Get dataset
X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff"
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    shuffle=True,
    stratify=y,
)
# endregion

# region Train rejector and classifier
estimator = MLPClassifier(
    hidden_layer_sizes=(40,),
    max_iter=20,
    alpha=1e-2,
    solver="adam",
    random_state=0,
    learning_rate_init=0.005,
    early_stopping=True,
    validation_fraction=0.11,
    n_iter_no_change=5,
    tol=1e-3,
    verbose=10,
)
pipe = make_pipeline(StandardScaler(), estimator)
rejector = ThresholdFallbackClassifier(pipe, threshold=0.8)
rejector.fit(X_train, y_train)
# endregion

# region Evaluate
rejector.set_params(fallback_mode="ignore")
print("MLPClassifier without fallbacks:")
print(f"Train accuracy: {rejector.score(X_train, y_train) * 100.0:.2f}%")
print(f"Test  accuracy: {rejector.score(X_test, y_test) * 100.0:.2f}%")

print("\nMLPClassifier with fallbacks:")
rejector.set_params(fallback_mode="return")
print(f"Train accuracy: {rejector.score(X_train, y_train) * 100.0:.2f}%")
print(f"Test  accuracy: {rejector.score(X_test, y_test) * 100.0:.2f}%")

PAConfusionMatrixDisplay.from_estimator(rejector, X_test, y_test)
# endregion

# region Generate random images and plot rejections
X_random = np.array([generate_random_combination(X) for _ in range(14_000)])
X_comb = np.concatenate([X_test, X_random])
y_score = rejector.predict_proba(X_comb)
fallback_mask = y_score.get_dense_fallback_mask()
scale = X_comb.max(axis=None)
plt.figure(figsize=(7, 7))
for i in range(30):
    plot = plt.subplot(5, 6, i + 1)
    plt.imshow(
        X_comb[fallback_mask][np.random.choice(fallback_mask.sum())].reshape(28, 28),
        interpolation="nearest",
        cmap=plt.cm.RdBu,
        vmin=-scale,
        vmax=scale,
    )
    plot.set_xticks(())
    plot.set_yticks(())
plt.suptitle("Examples of rejected digits")
plt.show()
# endregion
