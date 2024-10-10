Welcome to ``scikit-fallback``'s documentation!
=============================================

**scikit-fallback** (*scikit* - compatible with `scikit-learn <https://scikit-learn.org>`_,
*fallback* - abstention from taking actions) is a lightweight Python package for machine
learning with a reject option offering various tools to build estimators and scorers
supporting *fallbacks*, or *rejections*.

Assume that your domain can have:

* more classes than what your classifier was trained on (e.g., an unexpected buy_pizza
  intent encountered by your dialogue systems for bank applications);
* ambiguous examples (e.g., an image of both a cat and a dog passed to a cat-vs-dog
  classifier);
* classes with high misclassification costs (e.g., false-negatives in cancer diagnosis).

You might want to leverage additional experts like humans to tackle such anomalies.
``scikit-fallback`` can wrap your estimators and scorers, and also offer additional
objects to either predict fallback labels so that your pipelines hand the corresponding
samples off to other systems, or store fallback masks to evaluate the ability of your
pipelines to predict and reject correctly.


.. note::

   This project is under active development.


.. toctree::

   installation
   api
