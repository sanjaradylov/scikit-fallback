"""Extensions to numpy ndarrays supporting fallback masks."""

__all__ = (
    "fbarray",
    "FBNDArray",
)

import numpy as np

from scipy.sparse import coo_array

from ..utils._legacy import validate_params


class FBNDArray(np.ndarray):
    """Same as numpy ndarray but stores also additional fallback information.

    FBNDArrays are usually returned by ``predict``, ``predict_proba``, or
    ``predict_log_proba`` methods of fallback meta-estimators.

    Parameters
    ----------
    predictions : array-like of shape (n_samples,)
        An array of base-estimator predictions.
    fallback_mask : array-like of shape (n_samples,) or (1, n_samples), or None
        Array mask indicating whether i-th sample was rejected by a
        fallback meta-estimator. If None, defaults to all-zeros.
        Further stored as sparse arrays by the ``fallback_mask`` property.

    Examples
    --------
    >>> import numpy as np
    >>> from skfb.core import array as ska
    >>> y = np.array([0, 1, 0, 1, 0, 1])
    >>> f = [0, 0, 0, 0, 1, 1]
    >>> y_pred = ska.fbarray(y, f)
    >>> y_pred
    FBNDArray([0, 1, 0, 1, 0, 1])
    >>> y_pred.get_dense_fallback_mask()
    array([False, False, False, False,  True,  True])
    >>> y_pred.get_dense_neg_fallback_mask()
    array([ True,  True, False, False, False, False])
    >>> y_pred.fallback_rate
    0.3333333333333333
    >>> y_pred.as_comb(fallback_label=2)
    array([0, 1, 0, 1, 2, 2])
    """

    def __new__(cls, predictions, fallback_mask=None):
        obj = np.asarray(predictions).view(cls)
        obj._fallback_mask = cls._validate_fallback_mask(fallback_mask, len(obj))
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            empty = np.array([], dtype=np.bool_)
            self._fallback_mask = getattr(obj, "_fallback_mask", empty)

    @classmethod
    @validate_params(
        {
            "fallback_mask": ["array-like", None, list],
        },
        prefer_skip_nested_validation=True,
    )
    def _validate_fallback_mask(cls, fallback_mask, num_predictions):
        """Returns COO sparse ``fallback_mask`` if it's a valid prediction mask."""
        if fallback_mask is None or len(fallback_mask) == 0:
            return coo_array(np.array([], dtype=np.bool_))

        if len(fallback_mask) != num_predictions:
            raise ValueError(
                f"Mask size = {len(fallback_mask)} is greater than number of "
                f"elements of array = {num_predictions}"
            )
        else:
            return coo_array(np.asarray(fallback_mask, dtype=np.bool_))

    @property
    def fallback_mask(self):
        """Returns the sparse fallback mask."""
        return self._fallback_mask

    @fallback_mask.setter
    def fallback_mask(self, fallback_mask):
        """Sets a new sparse fallback mask."""
        self._fallback_mask = self._validate_fallback_mask(fallback_mask, len(self))

    @property
    def fallback_rate(self):
        """Calculates the sparsity of the fallback mask."""
        if len(self) == 0:
            return 0.0
        return self.fallback_mask.count_nonzero() / len(self)

    def get_dense_fallback_mask(self):
        """Converts ``fallback_mask`` to 1D ndarray."""
        mask = self.fallback_mask.toarray()
        if mask.ndim == 2:  # For scipy<1.13
            mask = mask[0]
        return mask

    def get_dense_neg_fallback_mask(self):
        """Returns negation of ``fallback_mask`` (acceptance mask) as 1D ndarray."""
        return ~self.get_dense_fallback_mask()

    def as_comb(self, fallback_label=-1):
        """Returns an ndarray of both predictions and fallbacks based on the mask."""
        y_comb = np.asarray(self).copy()
        y_comb[self.get_dense_fallback_mask()] = fallback_label
        return y_comb


def fbarray(predictions, fallback_mask=None):
    """Creates an ndarray of predictions that also stores fallback information."""
    return FBNDArray(predictions, fallback_mask=fallback_mask)
