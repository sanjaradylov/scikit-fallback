"""The :mod:`skfb.metrics` module includes score functions with a reject option."""

__all__ = (
    "fallback_quality_auc_score",
    "fallback_quality_curve",
    "get_scoring",
    "predict_accept_confusion_matrix",
    "predict_reject_accuracy_score",
    "predict_reject_recall_score",
    "prediction_quality",
    "FQCurveDisplay",
    "PAConfusionMatrixDisplay",
    "PairedHistogramDisplay",
)

from ._classification import (
    get_scoring,
    predict_accept_confusion_matrix,
    predict_reject_accuracy_score,
    predict_reject_recall_score,
)
from ._common import prediction_quality
from ._plot import FQCurveDisplay, PAConfusionMatrixDisplay, PairedHistogramDisplay
from ._ranking import fallback_quality_auc_score, fallback_quality_curve
