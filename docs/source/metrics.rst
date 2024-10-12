Combined Metrics
================

The ``skfb.metrics`` module includes score functions with a reject option and plotting
utilities.


Combined Classification Metrics
-------------------------------

.. autofunction:: skfb.metrics.predict_reject_accuracy_score

.. autofunction:: skfb.metrics.predict_reject_recall_score

.. autofunction:: skfb.metrics.predict_accept_confusion_matrix

.. autofunction:: skfb.metrics.fallback_quality_auc_score

.. autofunction:: skfb.metrics.fallback_quality_curve


Common Metrics
--------------

.. autofunction:: skfb.metrics.prediction_quality


Plotting Utilities
------------------

.. autoclass:: skfb.metrics.PAConfusionMatrixDisplay
    :inherited-members: plot, from_estimator, from_predictions

.. autoclass:: skfb.metrics.FQCurveDisplay
    :inherited-members: plot, from_estimator, from_predictions

.. autoclass:: skfb.metrics.PairedHistogramDisplay
    :inherited-members: plot, from_estimator, from_predictions
