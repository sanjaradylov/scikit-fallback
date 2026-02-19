Core Utilities
==============

The ``skfb.core.array`` module implements NDArrays for both rejection-aware and
rejection-unaware evaluation. They expand ``numpy`` NDArrays by adding sparse
and dense attributes masking fallbacks and collecting fallback statistics.

.. autoclass:: skfb.core.array.FBNDArray
    :members:

.. autoclass:: skfb.core.array.ENDArray
    :members:

The ``skfb.core.exceptions`` module defines core exceptions and warnings.

.. autoclass:: skfb.core.exceptions.SKFBException

.. autoclass:: skfb.core.exceptions.SKFBException
