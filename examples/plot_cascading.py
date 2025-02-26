"""
========================================
Model Cascading for Topic Classification
========================================

An example plot of :class:`skfb.ensemble.ThresholdCascadeClassifier`.
"""

# region Imports
from time import perf_counter

from matplotlib import pyplot as plt
import numpy as np

from skfb.ensemble import ThresholdCascadeClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import Normalizer

# endregion


random_state = np.random.RandomState(1142025)

# region Load & split train-test data
data_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=random_state)
data_test = fetch_20newsgroups(subset="test", shuffle=True, random_state=random_state)
y_names = data_train.target_names
X_train, y_train = data_train.data, data_train.target
X_test, y_test = data_test.data, data_test.target
# endregion

# region Define and train base estimators and cascade
weak = make_pipeline(
    CountVectorizer(min_df=32, max_features=1_536),
    CalibratedClassifierCV(
        MultinomialNB(),
        cv=4,
        n_jobs=2,
    ),
)
strong = make_pipeline(
    make_union(
        TfidfVectorizer(
            analyzer="char_wb",
            min_df=32,
            max_df=0.99,
            ngram_range=(1, 3),
        ),
        TfidfVectorizer(
            min_df=128,
            max_df=0.95,
            ngram_range=(1, 2),
        ),
    ),
    TruncatedSVD(n_components=512, random_state=42),
    Normalizer(copy=False),
    LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
        max_iter=200,
        cv=4,
        verbose=1,
        n_jobs=4,
    ),
)
cascade = ThresholdCascadeClassifier(
    [weak, strong],
    thresholds=0.7,
    response_method="predict_proba",
    return_earray=True,
    n_jobs=2,
    verbose=True,
)
cascade.fit(X_train, y_train)
# endregion


# region Print classification report
def report(estimator, X, y, title, target_names=None, is_cascade=False):
    print(f"[{title}]:")
    start = perf_counter()
    y_pred_ = estimator.predict(X)
    time_elapsed = perf_counter() - start
    print(f"Time: {time_elapsed:.2f} sec.")
    print(classification_report(y, y_pred_, target_names=target_names))

    if is_cascade:
        print(f"Acceptance rates: {y_pred_.acceptance_rates[0]:.2f}")

    print("=" * 20)

    return time_elapsed


report(cascade, X_test, y_test, "Cascade", target_names=y_names, is_cascade=True)
report(cascade.set_estimators(0), X_test, y_test, "Weak Model", target_names=y_names)
report(cascade.set_estimators(1), X_test, y_test, "Strong Model", target_names=y_names)
# endregion

# region Plot threshold-performance graph
WEAK_MODEL_COST, STRONG_MODEL_COST = 1.0, 6.0


def calculate_cost(acceptance_rates):
    """Calculates resources spent to infer w/ cascade."""
    return STRONG_MODEL_COST * acceptance_rates + WEAK_MODEL_COST * (
        1 - acceptance_rates
    )


cascade.reset_estimators()

thresholds = np.linspace(0.05, 0.95, 30)
scores = []
deferral_costs = []
for threshold in thresholds:
    y_pred = cascade.set_params(thresholds=threshold).predict(X_test)

    scores.append(accuracy_score(y_test, y_pred))

    deferral_cost = calculate_cost(y_pred.acceptance_rates[1])
    deferral_costs.append(deferral_cost)

plt.figure(figsize=(10, 6))
plt.plot(
    deferral_costs[5:-5],
    scores[5:-5],
    marker="o",
    linestyle="--",
    linewidth=1,
    markersize=2,
    color="b",
    label="Cost-Performance Curve",
)
plt.scatter(
    deferral_costs[0],
    scores[0],
    marker="*",
    color="g",
    s=130,
    label="Weak Model",
)
plt.scatter(
    deferral_costs[-1],
    scores[-1],
    marker="p",
    color="r",
    s=190,
    label="Strong Model",
)
plt.title("Model Cascading: Efficiency-Accuracy Tradeoff")
plt.xlabel("Inference Cost")
plt.ylabel("Accuracy Score")
plt.legend()
plt.show()
# endregion
