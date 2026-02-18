![PyPi](https://img.shields.io/pypi/v/scikit-fallback)
[![Downloads](https://static.pepy.tech/badge/scikit-fallback)](https://pepy.tech/project/scikit-fallback)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/sanjaradylov/scikit-fallback/badge)](https://www.codefactor.io/repository/github/sanjaradylov/scikit-fallback)
![Python package workflow](https://github.com/sanjaradylov/scikit-fallback/actions/workflows/python-package.yml/badge.svg)
[![Docs](https://github.com/sanjaradylov/scikit-fallback/actions/workflows/build-docs.yml/badge.svg)](https://github.com/sanjaradylov/scikit-fallback/actions/workflows/build-docs.yml)
[![PythonVersion](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3913/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fsanjaradylov%2Fscikit-fallback)](https://x.com/intent/tweet?text=Wow:%20https%3A%2F%2Fgithub.com%2Fsanjaradylov%2Fscikit-fallback%20@sanjaradylov)

# 👁 Overview

**`scikit-fallback`** is a `scikit-learn`-compatible Python package for selective machine learning.

## TL;DR

🔙 Augment your classification pipelines with
[`skfb.estimators`](https://scikit-fallback.readthedocs.io/en/latest/estimators.html#estimators)
such as
[`AnomalyFallbackClassifier`](https://scikit-fallback.readthedocs.io/en/latest/estimators.html#anomaly-based-fallback-classifiers)
and
[`ThresholdFallbackClassifier`](https://scikit-fallback.readthedocs.io/en/latest/estimators.html#skfb.estimators.ThresholdFallbackClassifier)
to allow them to *abstain* from predictions in cases of uncertanty or anomaly.<br>
📊 Inspect their performance by calculating *combined*, *prediction-rejection* metrics
such as [`predict_reject_recall_score`](https://scikit-fallback.readthedocs.io/en/latest/metrics.html#skfb.metrics.predict_reject_recall_score),
or visualizing distributions of confidence scores with
[`PairedHistogramDisplay`](https://scikit-fallback.readthedocs.io/en/latest/metrics.html#skfb.metrics.PairedHistogramDisplay),
and other tools from [`skfb.metrics`](https://scikit-fallback.readthedocs.io/en/latest/metrics.html#).<br>
🎶 Combine your costly ensembles with `RoutingClassifier` or in
`ThresholdCascadeClassifierCV` and other `skfb.ensemble` meta-estimators to streamline
inference while elevating model performance.<br>
📒See [documentation](https://scikit-fallback.readthedocs.io/en/latest/index.html),
[tutorials](https://medium.com/@sshadylov), and [examples](./examples/) for more details and motivation.


# 🤔 Why `scikit-fallback`?

To *fall back (on)* means to retreat from making predictions, to rely on other
tools for support. `scikit-fallback` offers functionality to enhance your machine learning
solutions with selectiveness and a reject option.

## Machine Learning with Rejections

To allow your classification pipelines to abstain from predictions, you can
wrap them with a *rejector*. Training a rejector means both fitting your model and
learning to accept or reject predictions. Evaluation of a rejector depends
on *fallback mode* (inference with or without *fallback labels*) and measures the ability
of the rejector to both accept correct predictions and reject ambiguous ones.

For example,
[`skfb.estimators.ThresholdFallbackClassifierCV`](https://scikit-fallback.readthedocs.io/en/latest/estimators.html#skfb.estimators.ThresholdFallbackClassifierCV)
fits a base estimator and then finds the best confidence threshold s.t. predictions w/
maximum probability lower that this are rejected:

```python
>>> import numpy as np
>>> from sklearn.linear_model import LogisticRegression
>>> from skfb.estimators import ThresholdFallbackClassifierCV
>>> X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2., 2.5]])
>>> y = np.array([0, 1, 0, 1, 0, 1])
>>> # Train LogisticRegression and let it fallback based on confidence scores.
>>> rejector = ThresholdFallbackClassifierCV(
...     estimator=LogisticRegression(random_state=0),
...     thresholds=(0.5, 0.55, 0.6, 0.65),
...     ambiguity_threshold=0.0,
...     cv=2,
...     fallback_label=-1,
...     fallback_mode="store").fit(X, y)
>>> # If probability is lower than this, predict `fallback_label` = -1.
>>> rejector.threshold_
0.55
>>> # Make predictions and see which inputs were accepted or rejected.
>>> y_pred = rejector.predict(X)
>>> # If `fallback_mode` == `"store", always accept but also mask rejections.
>>> y_pred, y_pred.get_dense_fallback_mask()
(FBNDArray([0, 1, 0, 1, 1, 1]),
    array([False, False, False, False,  True, False]))
>>> # This allows calculation of combined metrics (e.g., predict-reject accuracy).
>>> rejector.score(X, y)
1.0
>>> # Otherwise, allow fallbacks
>>> rejector.set_params(fallback_mode="return").predict(X)
array([ 0,  1,  0,  1, -1,  1])
>>> # and calculate accuracy only on accepted samples,
>>> rejector.score(X, y)
1.0
>>> # or just switch off rejections and fallback to a plain LogisticRegression.
>>> rejector.set_params(fallback_mode="ignore").score(X, y)
0.8333333333333334
```

See [Estimators](https://scikit-fallback.readthedocs.io/en/latest/estimators.html#) for
more examples of rejection meta-estimators and
[Combined Metrics](https://scikit-fallback.readthedocs.io/en/latest/metrics.html)
for evaluation and inspection tools.

## Ensembling

While common ensembling methods such as voting and stacking aim to boost predictive performance,
they also increase inference costs as a result of output aggregations. Alternatively, we could
learn to choose which individual model or subset of models in an ensemble should make a
decision.

For example, `skfb.ensemble.ThresholdCascadeClassifierCV` builds a *cascade* from a sequence of
models arranged by their inference costs (and basically, by their performance - e.g., from
weakest but fastest to strongest but slowest) and learns confidence thresholds that determine
whether the current model in the sequence makes a prediction or defers to the next model
based on its confidence score for a given input:

```python
>>> from skfb.ensemble import ThresholdCascadeClassifierCV
>>> from sklearn.datasets import make_classification
>>> from sklearn.ensemble import HistGradientBoostingClassifier
>>> X, y = make_classification(
...     n_samples=1_000, n_features=100, n_redundant=97, class_sep=0.1, flip_y=0.05,
...     random_state=0)
>>> weak = HistGradientBoostingClassifier(max_iter=10, max_depth=2, random_state=0)
>>> okay = HistGradientBoostingClassifier(max_iter=20, max_depth=3, random_state=0)
>>> buff = HistGradientBoostingClassifier(max_iter=99, max_depth=4, random_state=0)
>>> # Train all models and learn thresholds per model s.t. if the current model's max
>>> # confidence score is lower, it defers the decision to the next in the cascade.
>>> cascading = ThresholdCascadeClassifierCV(
...     estimators=[weak, okay, buff],
...     costs=[1.1, 1.2, 1.99],
...     cv_thresholds=5,
...     cv=3,
...     scoring="accuracy",
...     return_earray=True,
...     response_method="predict_proba").fit(X, y)
>>> # Best thresholds for `weak` and `okay`
>>> # (`buff` will always predict if `weak` and `okay` fall back):
>>> cascading.best_thresholds_
array([0.6125, 0.8375])
>>> # If `return_earray` is True, predictions will be of type `skfb.core.FBNDArray`,
>>> # which store `acceptance_rate` w/ the ratios of accepted inputs per model.
>>> cascading.predict(X).acceptance_rates
array([0.659, 0.003, 0.338])
```


# 🏗 Installation
`scikit-fallback` requires:
* Python (>= 3.10,< 3.14)
* scikit-learn (>=1.0)
* numpy
* scipy
* matplotlib (>=3.0) (optional)

and along with the requirements can be installed via `pip` :

```bash
pip install scikit-fallback
```


# 🔗 Links

1. [Documentation](https://scikit-fallback.readthedocs.io/en/latest/index.html)
2. [Medium Series](https://medium.com/@sshadylov)
3. Examples & Notebooks: [examples/](./examples/) and https://kaggle.com/sshadylov
4. Related Research:
   1. Hendrickx, K., Perini, L., Van der Plas, D. et al. Machine learning with a reject option: a survey. Mach Learn 113, 3073–3110 (2024).
   2. Wittawat Jitkrittum, Neha Gupta, Aditya K Menon, Harikrishna Narasimhan, Ankit Rawat, and Sanjiv Kumar. When does confidence-based cascade deferral suffice? NeurIPS, 36, 2024.
   3. And more (coming soon).
