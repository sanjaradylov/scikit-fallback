Welcome to ``scikit-fallback``'s documentation!
===============================================

🎯 **Build Adaptive Pipelines: Orchestrate Models with Selective Prediction!**

**scikit-fallback** is a scikit-learn-compatible Python package for *selective machine learning*.
It lets you *orchestrate multiple classifiers with fallback strategies*, routing uncertain or anomalous samples
to specialized models, human experts, or fallback handlers. Perfect for enabling reliable and intelligent decisions
in high-stakes domains.

.. image:: https://img.shields.io/pypi/v/scikit-fallback?logo=pypi&logoColor=white
   :alt: PyPI Version
   :target: https://pypi.org/project/scikit-fallback/

.. image:: https://img.shields.io/badge/Python-3.9+-2E86AB?logo=python&logoColor=white
   :alt: Python 3.9+
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/scikit--learn-Compatible-F7931E?logo=scikit-learn&logoColor=white
   :alt: scikit-learn Compatible

.. image:: https://img.shields.io/badge/License-BSD--3-green?logo=opensourceinitiative&logoColor=white
   :alt: BSD-3 License
   :target: https://github.com/sanjaradylov/scikit-fallback/blob/main/LICENSE

.. image:: https://img.shields.io/github/stars/sanjaradylov/scikit-fallback?logo=github&logoColor=black
   :alt: GitHub Stars
   :target: https://github.com/sanjaradylov/scikit-fallback

.. image:: https://static.pepy.tech/badge/scikit-fallback
   :alt: Downloads
   :target: https://pepy.tech/project/scikit-fallback


Why Fallbacks? 🤔
-----------------

To *fall back (on)* means to retreat from making predictions, to rely on other tools for support.
**scikit-fallback** flips the paradigm of blind and uncontrolled predictions and offers functionality
to enhance your machine learning solutions with selectiveness and a reject option:

- 🤷‍♂️ *Reject ambiguous predictions and reduce costly misclassifications* (confidence < threshold; classifier + outlier detector)
- 🧠 *Wrap your pipelines with rejectors tidily* instead of handcrafting rejections out-of-pipeline
- 🧮 *Measure combined metrics* to understand how successful your model in acceptance and rejection is
- 🔀 *Choose only appropriate models* from ensembles for optimal performance-efficiency tradeoff
- 🔎 *Track model decisions* to see which samples a model rejected / accepted

*Real-world scenarios* where this matters:

- 💳 *Finance*: Fraud model ➡️ detect ambiguous transaction ➡️ escalate for manual review
- 🤖 *Dialogue*: Intent classifier ➡️ prefer smaller specialist LLM ➡️ route to generate response
- 🏥 *Medical*: Disease detector ➡️ reject uncertain prediction ➡️ defer to human doctor


Key Features ✨
---------------

**Rejection:**
  *Wrap any scikit-learn classifier with a reject option:*

  * Confidence threshold rejection (abstain when uncertain)
  * Per-class thresholds
  * Custom rule-based logic
  * Anomaly detection for deferral

**Ensembling:**
  *Combine multiple models intelligently:*

  * Semantic routing (select best model for each sample)
  * Threshold cascades (model pipeline with early rejection)
  * Track which model made each prediction

**Metrics:**
  *Evaluate abstention and classification performance as combined metrics:*

  * Acceptance/rejection confusion matrices
  * Accept/reject accuracy decompositions
  * Ranking metrics with fallback support


Quick Start 🚀
--------------

Use a *rejector* to grant your classifier a reject option:

.. code-block:: python

   >>> import numpy as np
   >>> from sklearn.linear_model import LogisticRegression
   >>> from skfb.estimators import ThresholdFallbackClassifierCV
   >>> X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2., 2.5]])
   >>> y = np.array([0, 1, 0, 1, 0, 1])
   >>> # Train LogisticRegression and let it fallback based on confidence scores.
   >>> rejector = ThresholdFallbackClassifierCV(
   ...     estimator=LogisticRegression(random_state=0),
   ...     thresholds=(0.5, 0.55, 0.6, 0.65),
   ...     ambiguity_threshold=0.0,
   ...     cv=2,
   ...     fallback_label=-1,
   ...     fallback_mode="store").fit(X, y)
   >>> # If probability is lower than this, predict `fallback_label` = -1.
   >>> rejector.threshold_
   0.55
   >>> # Make predictions and see which inputs were accepted or rejected.
   >>> y_pred = rejector.predict(X)
   >>> # If `fallback_mode` == `"store", always accept but also mask rejections.
   >>> y_pred, y_pred.get_dense_fallback_mask()
   (FBNDArray([0, 1, 0, 1, 1, 1]),
      array([False, False, False, False,  True, False]))
   >>> # This allows calculation of combined metrics (e.g., predict-reject accuracy).
   >>> rejector.score(X, y)
   1.0
   >>> # Otherwise, allow fallbacks
   >>> rejector.set_params(fallback_mode="return").predict(X)
   array([ 0,  1,  0,  1, -1,  1])
   >>> # and calculate accuracy only on accepted samples,
   >>> rejector.score(X, y)
   1.0
   >>> # or just switch off rejections and fallback to a plain LogisticRegression.
   >>> rejector.set_params(fallback_mode="ignore").score(X, y)
   0.8333333333333334
   >>>


Or use a *router* for multi-stage model routing:

.. code-block:: python

   >>> from skfb.ensemble import ThresholdCascadeClassifierCV
   >>> from sklearn.datasets import make_classification
   >>> from sklearn.ensemble import HistGradientBoostingClassifier
   >>> X, y = make_classification(
   ...     n_samples=1_000, n_features=100, n_redundant=97, class_sep=0.1, flip_y=0.05,
   ...     random_state=0)
   >>> weak = HistGradientBoostingClassifier(max_iter=10, max_depth=2, random_state=0)
   >>> okay = HistGradientBoostingClassifier(max_iter=20, max_depth=3, random_state=0)
   >>> buff = HistGradientBoostingClassifier(max_iter=99, max_depth=4, random_state=0)
   >>> # Train all models and learn thresholds per model s.t. if the current model's max
   >>> # confidence score is lower, it defers the decision to the next in the cascade.
   >>> cascading = ThresholdCascadeClassifierCV(
   ...     estimators=[weak, okay, buff],
   ...     costs=[1.1, 1.2, 1.99],
   ...     cv_thresholds=5,
   ...     cv=3,
   ...     scoring="accuracy",
   ...     return_earray=True,
   ...     response_method="predict_proba").fit(X, y)
   >>> # Best thresholds for `weak` and `okay`
   >>> # (`buff` will always predict if `weak` and `okay` fall back):
   >>> cascading.best_thresholds_
   array([0.6125, 0.8375])
   >>> # If `return_earray` is True, predictions will be of type `skfb.core.FBNDArray`,
   >>> # which store `acceptance_rate` w/ the ratios of accepted inputs per model.
   >>> cascading.predict(X).acceptance_rates
   array([0.659, 0.003, 0.338])


And see :doc:`api` for more information.


Documentation 📚
----------------

.. toctree::
   :maxdepth: 2

   Home <self>
   installation
   api


Learn More ⛓️
-------------

- 🐙 *Code:* Follow `Github Repository <https://github.com/sanjaradylov/scikit-fallback>`_ for implementations, discussions, and updates
- 📚 *Full Guide:* See :doc:`api` for estimators, metrics, and ensemble strategies
- 📝 *Blog Series:* Check out the `Kaggle <https://kaggle.com/sshadylov>`_ and
  `Medium tutorials <https://medium.com/@sshadylov>`_ for deeper dives
- 💻 *Examples:* Browse `Examples <https://github.com/sanjaradylov/scikit-fallback/tree/main/examples>`_
  for rejection analysis, cascading, and other demos

.. note::

   **Status:** v0.2.0 stable release with production-ready APIs. Active development underway!


Inspiration & References 📖
---------------------------

**scikit-fallback** builds on decades of research in selective classification
and rejection. Some inspirations include:

.. [1] Chow, C. K.
   *On optimum recognition error and reject tradeoff.*
   IEEE Transactions on Information Theory 16, no. 1 (1970).
   https://i.org/10.1109/TIT.1970.1054406

.. [2] Wittawat Jitkrittum, Neha Gupta, Aditya K Menon, Harikrishna Narasimhan, Ankit Rawat, Sanjiv Kumar.
   *When does confidence-based cascade deferral suffice?*
   Advances in Neural Information Processing Systems (NeurIPS) (2023).
   https://arxiv.org/abs/2307.02764

.. [3] Kilian Hendrickx, Lorenzo Perini, Dries Van der Plas, Wannes Meert & Jesse Davis.
   *Machine learning with a reject option: a survey.*
   Machine Learning 113, 3073–3110 (2024).
   https://doi.org/10.1007/s10994-024-06534-x
