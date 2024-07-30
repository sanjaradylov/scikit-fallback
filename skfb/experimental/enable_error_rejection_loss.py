"""Enables `skfb.metrics.error_rejection_loss`."""

from .. import metrics
from ..metrics._classification import error_rejection_loss


setattr(metrics, "error_rejection_loss", error_rejection_loss)
metrics.__all__ += ("error_rejection_loss",)
