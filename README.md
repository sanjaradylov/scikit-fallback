![PyPi](https://img.shields.io/pypi/v/scikit-fallback)
[![Downloads](https://static.pepy.tech/badge/scikit-fallback)](https://pepy.tech/project/scikit-fallback)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/sanjaradylov/scikit-fallback/badge)](https://www.codefactor.io/repository/github/sanjaradylov/scikit-fallback)
![Python package workflow](https://github.com/sanjaradylov/scikit-fallback/actions/workflows/python-package.yml/badge.svg)
[![PythonVersion](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/release/python-3913/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fsanjaradylov%2Fscikit-fallback)](https://x.com/intent/tweet?text=Wow:%20https%3A%2F%2Fgithub.com%2Fsanjaradylov%2Fscikit-fallback%20@sanjaradylov)
[![Twitter Follow](https://img.shields.io/twitter/follow/sanjaradylov?style=social)](https://twitter.com/sanjaradylov)

**scikit-fallback** is a scikit-learn-compatible Python package for machine learning
with a reject option.

### 👩‍💻 Usage

To allow your probabilistic pipeline to *fallback*—i.e., abstain from predictions—you can
wrap it with a `skfb` *rejector*. Training a rejector means both fitting your model and
learning to accept or reject predictions. Evaluation of a rejector depends
on *fallback mode* (inference with or without *fallback labels*) and measures the ability
of the rejector to both accept correct predictions and reject ambiguous ones.

For example, `skfb.estimators.ThresholdFallbackClassifierCV` fits the base estimator and then
finds the best confidence threshold via cross-validation. If `fallback_mode == "store"`, then the
rejector returns `skfb.core.array.FBNDArray` of predictions and a sparse fallback-mask property,
which lets us summarize the accuracy of both predictions and rejections.

```python
from skfb.estimators import ThresholdFallbackClassifierCV
from sklearn.linear_model import LogisticRegressionCV

rejector = ThresholdFallbackClassifierCV(
    LogisticRegressionCV(cv=4, random_state=0),
    thresholds=10,
    ambiguity_threshold=0.05,
    cv=5,
    fallback_label=-1,
    fallback_mode="store",
)
rejector.fit(X_train, y_train)  # Train base estimator and learn best threshold
rejector.score(X_test, y_test)  # Compute acceptance-correctness accuracy score
```

And for more motivation and information on usage, please visit our
[documentation](https://scikit-fallback.readthedocs.io/en/latest/index.html)
and refer to the
[Medium series](https://medium.com/@sshadylov) on machine learning with a reject option.


### 🏗 Installation
`scikit-fallback` requires:
* Python (>=3.9,<3.13)
* scikit-learn (>=1.0)
* numpy
* scipy
* matplotlib (>=3.0) (optional)

and along with the requirements can be installed via `pip` :

```bash
pip install -U scikit-fallback
```


### 📚 Examples

See the [`examples/`]([examples/](https://github.com/sanjaradylov/scikit-fallback/tree/main/examples))
directory for various applications of fallback estimators
and scorers to scikit-learn-compatible pipelines.

### 🔗 References

1. Hendrickx, K., Perini, L., Van der Plas, D. et al. Machine learning with a reject option: a survey. Mach Learn 113, 3073–3110 (2024). https://doi.org/10.1007/s10994-024-06534-x
