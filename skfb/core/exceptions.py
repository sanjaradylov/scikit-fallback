"""Exceptions raised by scikit-fallback estimators."""

__all__ = ("SKFBException", "SKFBWarning")


class SKFBException(Exception):
    """Base exception for scikit-fallback estimators."""


class SKFBWarning(UserWarning):
    """Base warning for scikit-fallback estimators."""
