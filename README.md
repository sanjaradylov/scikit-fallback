![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

**scikit-fallback** is a scikit-learn-compatible Python package for machine learning
with a reject option.

## Get started w/ `scikit-fallback`

### Usage
```python
from skfb.estimators import RateFallbackClassifier
from skfb.metrics import
from sklearn.linear_model import LogisticRegression

rejector = RateFallbackClassifier(LogisticRegression(), fallback_rate=0.05, cv=5)
rejector.fit(X_train, y_train)
y_pred = rejector.predict(X_test)
print(predict_reject_accuracy_score(y_test, y_pred))
```

### Installation
`scikit-fallback` requires:
* scikit-learn (>=1.0)
* matplotlib (>=1.0)

### Examples

See the `examples/` directory for various applications of fallback estimators and
scorers to scikit-learn-compatible pipelines.
