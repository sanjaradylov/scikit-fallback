Installing `scikit-fallback`!
=============================

**scikit-fallback** supports Python 3.9+ and depends on ``scikit-learn>=1.0``, `numpy`,
and ``scipy``. An optional requirement is ``matplotlib`` (for visualization of metrics
from ``skfb.metrics``).

Install with ``pip``
--------------------

Usually these dependencies all pre-installed in any ML-powered
environment; otherwise, **scikit-fallback** tries installing newer versions of the
dependencies along with itself:::

    pip install -U scikit-fallback

.. note::

   You might encounter warnings from **scikit-fallback** if you have
   ``scikit-learn<1.3``. They are about several private features of ``scikit-learn``
   and shouldn't affect performance and API. Also, note that some older version of
   ``scikit-learn`` don't support ``numpy~=2.0``.


Build from Source
-----------------

To build **scikit-fallback** from source, clone the project, set up an environment,
install the package and all the dependencies:::


    git clone https://github.com/sanjaradylov/scikit-fallback.git
    cd scikit-fallback
    # Your environment activation here
    python -m pip install -e ".[tests]"
    pre-commit run --all-files
