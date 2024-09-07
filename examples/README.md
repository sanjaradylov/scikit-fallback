General examples
================

This is the gallery of examples that showcase how `scikit-fallback` can be combined with
`scikit-learn` and other packages to build and analyze machine-learning classifiers with
a reject option.

* [notebooks/osvm-mnist.ipynb](./notebooks/osvm-mnist.ipynb): combine scikit-learn's `OneClassSVM`
  and `MLPClassifier` with scikit-fallback's `AnomalyFallbackClassifier` to predict or reject
  noisy MNIST data.
* [plot_confusion_matrix.py](./plot_confusion_matrix.py): learn to reject low-confidence
  MNIST images and plot predict-accept confusion matrix.
* [plot_decision_boundary.py](./plot_decision_boundary.py): plot the regions where
  samples from `make_moons` are rejected by a simple `ThresholdFallbackClassifier`.
