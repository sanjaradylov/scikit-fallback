Welcome to ``scikit-fallback``'s documentation!
===============================================

**scikit-fallback** (*scikit* - compatible with `scikit-learn <https://scikit-learn.org>`_,
*fallback* - abstention from taking actions) is a lightweight Python package for machine
learning with a reject option offering various tools to build estimators and scorers
supporting *fallbacks*, or *rejections*.


Motivaion
---------

There are several reasons why rejections become essential. Assume that your domain can have:

* *more classes than what your classifier was trained* on (e.g., an unexpected buy_pizza
  intent encountered by your dialogue systems for bank applications);
* *ambiguous examples* (e.g., an image of both a cat and a dog passed to a cat-vs-dog
  classifier);
* *classes with high misclassification costs* (e.g., false-negatives in cancer diagnosis).

You might want to leverage additional experts like humans to tackle such anomalies.
``scikit-fallback`` can wrap your estimators and scorers, and also offer additional
objects to either predict fallback labels so that your pipelines hand the corresponding
samples off to other systems, or store fallback masks to evaluate the ability of your
pipelines to predict and reject correctly.


.. note::

   This project is under active development.


Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   api


References
----------

Research in machine learning with rejections and selective classification is evolving,
and there are many valuable works available online including from top conferences like
NeurIPS and ICML. Some of the components of **scikit-fallback** were inspired by the
following survey:

.. [1] Hendrickx, K., Perini, L., Van der Plas, D. et al.
   *Machine learning with a reject option: a survey.* Mach Learn 113, 3073–3110 (2024).
   https://doi.org/10.1007/s10994-024-06534-x

Also, take a look at the `Medium series <https://medium.com/@sshadylov>`_ on machine
learning with a reject option for motivation and information on usage.
