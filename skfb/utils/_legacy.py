"""Importing utilities of sklearn>=1.3, preventing errors of <1.3"""

__all__ = (
    "_fit_context",
    "validate_params",
)

import functools
import inspect
import warnings

try:
    # pylint: disable=no-name-in-module
    from sklearn.base import _fit_context
except ImportError:
    warnings.warn(
        "Fit methods within context managers aren't supported for sklearn<=1.2.",
        category=ImportWarning,
    )

    # pylint: disable=unused-argument
    def _fit_context(*, prefer_skip_nested_validation=False):
        """For sklearn<=1.2, fit methods aren't run within context managers."""

        def decorator(fit_method):

            @functools.wraps(fit_method)
            def wrapper(estimator, *args, **kwargs):
                return fit_method(estimator, *args, **kwargs)

            return wrapper

        return decorator


# pylint: disable=import-error,no-name-in-module
from sklearn.utils._param_validation import validate_params as _validate_params


if (
    "prefer_skip_nested_validation"
    not in inspect.signature(_validate_params).parameters
):
    warnings.warn(
        "Nested parameter validation isn't supported for sklearn<=1.2.",
        category=ImportWarning,
    )

    # pylint: disable=unused-argument
    def validate_params(parameter_constraints, *, prefer_skip_nested_validation):
        """For sklearn<=1.2, param validation is not nested."""
        # pylint: disable=missing-kwoa
        return _validate_params(parameter_constraints)

else:
    validate_params = _validate_params
