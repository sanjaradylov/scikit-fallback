"""Array validation utilities (mainly for indexing support)."""

from sklearn.base import check_array
from sklearn.utils import check_X_y


def check_X_y_sample_weight(X, y=None, sample_weight=None):
    """Validation of input data for unified indexing."""
    if y is not None:
        try:
            X, y = check_X_y(
                X,
                y,
                accept_sparse=True,
                accept_large_sparse=True,
                dtype=None,
                order=None,
                copy=False,
                ensure_all_finite=False,
                ensure_2d=False,
                allow_nd=True,
            )
        except TypeError:
            # `ensure_all_finite` is in place of `force_all_finite`
            X, y = check_X_y(
                X,
                y,
                accept_sparse=True,
                accept_large_sparse=True,
                dtype=None,
                order=None,
                copy=False,
                force_all_finite=False,
                ensure_2d=False,
                allow_nd=True,
            )
    else:
        try:
            X = check_array(
                X,
                accept_sparse=True,
                accept_large_sparse=True,
                dtype=None,
                order=None,
                copy=False,
                ensure_all_finite=False,
                ensure_2d=False,
                allow_nd=True,
            )
        except TypeError:
            # `ensure_all_finite` is in place of `force_all_finite`
            X = check_array(
                X,
                accept_sparse=True,
                accept_large_sparse=True,
                dtype=None,
                order=None,
                copy=False,
                force_all_finite=False,
                ensure_2d=False,
                allow_nd=True,
            )

    if sample_weight is not None:
        sample_weight = check_array(
            sample_weight,
            accept_sparse=False,
            ensure_2d=False,
            dtype=None,
            order="C",
            copy=False,
        )

    return X, y, sample_weight
