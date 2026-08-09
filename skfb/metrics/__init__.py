"""The :mod:`skfb.metrics` module includes score functions with a reject option."""

__all__ = (
    "FQCurveDisplay",
    "PAConfusionMatrixDisplay",
    "PairedHistogramDisplay",
    "UCCurveDisplay",
    "fallback_quality_auc_score",
    "fallback_quality_curve",
    "get_scoring",
    "oracle_auc_score",
    "oracle_curve",
    "oracle_utility_gap_score",
    "predict_accept_confusion_matrix",
    "predict_reject_accuracy_score",
    "predict_reject_recall_score",
    "prediction_quality",
    "utility_coverage_auc_score",
    "utility_coverage_curve",
)

from ._classification import (
    get_scoring,
    oracle_auc_score,
    oracle_curve,
    oracle_utility_gap_score,
    predict_accept_confusion_matrix,
    predict_reject_accuracy_score,
    predict_reject_recall_score,
)
from ._common import (
    prediction_quality,
    utility_coverage_auc_score,
    utility_coverage_curve,
)
from ._plot import (
    FQCurveDisplay,
    PAConfusionMatrixDisplay,
    PairedHistogramDisplay,
    UCCurveDisplay,
)
from ._ranking import fallback_quality_auc_score, fallback_quality_curve
