"""Base classes for estimators w/ a rejection option."""

__all__ = (
    "is_rejector",
    "RejectorMixin",
)


class RejectorMixin:
    """Mixin class for estimators w/ a rejection option."""

    _estimator_type = "rejector"

    def _more_tags(self):
        """For now, rejection-based learning is supervised."""
        return {"requires_y": True}


def is_rejector(estimator):
    """Returns True if ``estimator`` provides a rejection option."""
    return getattr(estimator, "_estimator_type") == "rejector"
