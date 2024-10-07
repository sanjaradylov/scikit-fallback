"""Importing utilities of sklearn>=1.2, preventing errors of <1.2.

NOTE: Dynamically creating private sklearn objects absent in older sklearn versions.
      This could be prevented with reimplementation of private sklearn validators
      but everything works fine for older sklearn versions - dummy validators are
      created and validation is ignored.
"""

__all__ = (
    "_fit_context",
    "validate_params",
)

import functools
import inspect
import sys
import warnings


# region Try importing ``_fit_context``
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
            """Simply returns wrapped method of estimator."""

            @functools.wraps(fit_method)
            def wrapper(estimator, *args, **kwargs):
                """Simply calls the method."""
                return fit_method(estimator, *args, **kwargs)

            return wrapper

        return decorator


# endregion

# region Try importing parameter validation
dummy_constraints = (
    "HasMethods",
    "Integral",
    "Interval",
    "Real",
    "StrOptions",
)

try:
    # pylint: disable=unused-import
    from sklearn.utils._param_validation import (
        HasMethods,
        Integral,
        Interval,
        Real,
        StrOptions,
        validate_params,
    )

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

        # pylint: disable=function-redefined,unused-argument
        def validate_params(parameter_constraints, *, prefer_skip_nested_validation):
            """For sklearn<=1.2, param validation is not nested."""
            # pylint: disable=missing-kwoa
            return _validate_params(parameter_constraints)

    else:
        validate_params = _validate_params

except ModuleNotFoundError:
    warnings.warn(
        "Parameter validation isn't supported for sklearn<1.2",
        category=ImportWarning,
    )

    class _Constraint:
        def __init__(self, *args, **kwargs):
            pass

    module = sys.modules[__name__]

    for constraint in dummy_constraints:
        Class = type(constraint, (_Constraint,), {"__slots__": ()})
        setattr(module, constraint, Class)

    # pylint: disable=unused-argument
    def validate_params(parameter_constraints, **validation_kw):
        """For sklearn<1.2, parameter validation isn't defined."""

        def decorator(function):
            """Simply returns wrapped function."""

            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                """Simply calls the function."""
                return function(*args, **kwargs)

            return wrapper

        return decorator

finally:
    __all__ += dummy_constraints
# endregion
