![PyPi](https://img.shields.io/pypi/v/scikit-fallback)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
[![PythonVersion](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/release/python-3913/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

**scikit-fallback** is a scikit-learn-compatible Python package for machine learning
with a reject option.

## Get started w/ `scikit-fallback`

### Installation
`scikit-fallback` requires:
* Python (>=3.9,< 3.13)
* scikit-learn (>=1.3)

```bash
pip install -U scikit-fallback
```

### Usage
```python
from skfb.estimators import RateFallbackClassifierCV
from skfb.metrics import predict_reject_accuracy_score
from sklearn.linear_model import LogisticRegression

rejector = RateFallbackClassifierCV(
    LogisticRegression(),
    fallback_rates=[0.05, 0.07],
    cv=5,
)
rejector.fit(X_train, y_train)
y_pred = rejector.predict(X_test)
print(predict_reject_accuracy_score(y_test, y_pred))
```

### Examples

See the `examples/` directory for various applications of fallback estimators and
scorers to scikit-learn-compatible pipelines.
