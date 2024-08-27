"""Fallback-based visualizations"""

from sklearn.metrics import accuracy_score, auc, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import check_is_fitted

from ..estimators.base import is_rejector
from ._classification import predict_accept_confusion_matrix
from ._ranking import fallback_quality_curve


def check_matplotlib_support(caller_name):
    """Raise ImportError with detailed error message if mpl is not installed.

    Plot utilities like any of the Display's plotting functions should lazily import
    matplotlib and call this helper before any computation.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires matplotlib.

    References:
        sklearn.utils._optional_dependencies.check_matplotlib_support
    """
    try:
        # pylint: disable=import-outside-toplevel,unused-import
        import matplotlib
    except ImportError as e:
        raise ImportError(
            f"{caller_name} requires matplotlib. You can install matplotlib with "
            "`pip install matplotlib`"
        ) from e


class PAConfusionMatrixDisplay(ConfusionMatrixDisplay):
    """Predict-Accept Confusion Matrix visualization.

    It is recommend to use
    :func:`~skfb.metrics.PAConfusionMatrixDisplay.from_estimator` or
    :func:`~skfb.metrics.PAConfusionMatrixDisplay.from_predictions` to
    create a :class:`PAConfusionMatrixDisplay`. All parameters are stored as
    attributes.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (2, 2)
        Predict-accept confusion matrix.
    display_labels : ndarray of shape (n_classes,), default=("No", "Yes")
        Display labels for plot. If None, display labels are set from 0 to
        `n_classes - 1`.
    fallback_rate : float, default=None
        Ratio of rejected samples to all samples.

    See Also
    --------
    skfb.metrics.predict_accept_confusion_matrix : Compute Confusion Matrix to evaluate
        the quality of predictions vs fallbacks.
    skfb.metrics.PAConfusionMatrixDisplay.from_estimator : Plot the confusion matrix
        given an estimator, the data, and the label.
    skfb.metrics.PAConfusionMatrixDisplay.from_predictions : Plot the confusion matrix
        given the true and predicted labels.
    sklearn.metrics.ConfusionMatrixDisplay : We inherit this class and adapt its
        methods to rejections.

    Notes
    -----
        Adapted from :class:`~sklearn.metrics.ConfusionMatrixDisplay`.
    """

    def __init__(
        self,
        confusion_matrix,
        *,
        display_labels=("No", "Yes"),
        fallback_rate=None,
    ):
        super().__init__(confusion_matrix, display_labels=display_labels)

        self.fallback_rate = fallback_rate

    def plot(
        self,
        *,
        include_values=True,
        cmap="viridis",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
        colorbar=True,
        im_kw=None,
    ):
        """Plots predict-accept confusion matrix.

        Same as :class:`~sklearn.metrics.ConfusionMatrixDisplay.plot` except that
        changes the label names.

        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is 'd' or '.2g' whichever is shorter.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.
        im_kw : dict, default=None
            Dict with keywords passed to `matplotlib.pyplot.imshow` call.

        Returns
        -------
        display : :class:`~skfb.metrics.PAConfusionMatrixDisplay`
            Returns a :class:`~sfkb.metrics.PAConfusionMatrixDisplay` instance
            that contains all the information to plot the confusion matrix.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from skfb.core import array as ska
        >>> from skfb.metrics import predict_accept_confusion_matrix
        >>> from skfb.metrics import PAConfusionMatrixDisplay
        >>> y_true =    np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
        >>> y_pred = ska.fbarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        ...                      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        >>> cm = predict_accept_confusion_matrix(y_true=y_true, y_pred=y_pred)
        >>> PAConfusionMatrixDisplay(cm).plot()
        <...>
        >>> plt.show()
        """
        super().plot(
            include_values=include_values,
            cmap=cmap,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            ax=ax,
            colorbar=colorbar,
            im_kw=im_kw,
        )

        self.ax_.set_xlabel("Accepted?")
        self.ax_.set_ylabel("Predicted correctly?")

        title = "Predict-Accept Confusion Matrix"
        if self.fallback_rate is not None:
            title += f"\n(fallback rate = {self.fallback_rate * 100.0:.2f}%)"

        self.ax_.set_title(title)

        return self

    @classmethod
    # pylint: disable=arguments-renamed
    def from_estimator(
        cls,
        rejector,
        X,
        y,
        *,
        labels=None,
        sample_weight=None,
        normalize=None,
        display_labels=None,
        include_values=True,
        xticks_rotation="horizontal",
        values_format=None,
        cmap="viridis",
        ax=None,
        colorbar=True,
        im_kw=None,
    ):
        """Plots PA Confusion Matrix given a rejector and some data.

        Parameters
        ----------
        estimator : rejector instance
            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last rejector is a classifier.
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.
        y : FBNDArray of shape (n_samples,)
            Target values.
        labels : array-like of shape (n_classes,), default=None
            List of labels to index the confusion matrix. This may be used to
            reorder or select a subset of labels. If `None` is given, those
            that appear at least once in `y_true` or `y_pred` are used in
            sorted order.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        normalize : {'true', 'pred', 'all'}, default=None
            Either to normalize the counts display in the matrix:
            - if `'true'`, the confusion matrix is normalized over the true
              conditions (e.g. rows);
            - if `'pred'`, the confusion matrix is normalized over the
              predicted conditions (e.g. columns);
            - if `'all'`, the confusion matrix is normalized by the total
              number of samples;
            - if `None` (default), the confusion matrix will not be normalized.
        display_labels : array-like of shape (n_classes,), default=None
            Target names used for plotting. By default, `labels` will be used
            if it is defined, otherwise the unique labels of `y_true` and
            `y_pred` will be used.
        include_values : bool, default=True
            Includes values in confusion matrix.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`, the
            format specification is 'd' or '.2g' whichever is shorter.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.
        im_kw : dict, default=None
            Dict with keywords passed to `matplotlib.pyplot.imshow` call.

        Returns
        -------
        display : :class:`~skfb.metrics.PAConfusionMatrixDisplay`

        See Also
        --------
        PAConfusionMatrixDisplay.from_predictions

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skfb.estimators import ThresholdFallbackClassifier
        >>> from skfb.metrics import PAConfusionMatrixDisplay
        >>> X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2., 2.5]])
        >>> y = np.array([0, 1, 0, 1, 0, 1])
        >>> estimator = LogisticRegression(random_state=0)
        >>> rejector = ThresholdFallbackClassifier(estimator, threshold=0.6).fit(X, y)
        >>> PAConfusionMatrixDisplay.from_estimator(rejector, X, y)
        <...>
        >>> plt.show()
        """
        method_name = f"{cls.__name__}.from_estimator"
        check_matplotlib_support(method_name)

        if not is_rejector(rejector):
            raise ValueError(f"{method_name} only supports rejectors")

        if isinstance(rejector, Pipeline):
            rejector_ = rejector[-1]
        else:
            rejector_ = rejector

        if rejector_.fallback_mode == "return":
            y_pred = rejector_.set_params(fallback_mode="store").predict(X)
            rejector_.set_params(fallback_mode="return")
        elif rejector_.fallback_mode == "ignore":
            y_pred = rejector_.set_params(fallback_mode="store").predict(X)
            rejector_.set_params(fallback_mode="ignore")
        else:
            y_pred = rejector.predict(X)

        return cls.from_predictions(
            y,
            y_pred,
            sample_weight=sample_weight,
            labels=labels,
            normalize=normalize,
            display_labels=display_labels,
            include_values=include_values,
            cmap=cmap,
            ax=ax,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            colorbar=colorbar,
            im_kw=im_kw,
        )

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        labels=None,
        sample_weight=None,
        normalize=None,
        display_labels=("No", "Yes"),
        include_values=True,
        xticks_rotation="horizontal",
        values_format=None,
        cmap="viridis",
        ax=None,
        colorbar=True,
        im_kw=None,
    ):
        """Plots PA Confusion Matrix given true and predicted labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.
        y_pred : FBNDArray (n_samples,)
            The predicted labels w/ the fallback mask given by the method `predict`
            of a rejector
        labels : array-like of shape (n_classes,), default=None
            List of labels to index the confusion matrix. This may be used to
            reorder or select a subset of labels. If `None` is given, those
            that appear at least once in `y_true` or `y_pred` are used in
            sorted order.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        normalize : {'true', 'pred', 'all'}, default=None
            Either to normalize the counts display in the matrix:
            - if `'true'`, the confusion matrix is normalized over the true
              conditions (e.g. rows);
            - if `'pred'`, the confusion matrix is normalized over the
              predicted conditions (e.g. columns);
            - if `'all'`, the confusion matrix is normalized by the total
              number of samples;
            - if `None` (default), the confusion matrix will not be normalized.
        display_labels : array-like of shape (n_classes,), default=None
            Target names used for plotting. By default, `labels` will be used
            if it is defined, otherwise the unique labels of `y_true` and
            `y_pred` will be used.
        include_values : bool, default=True
            Includes values in confusion matrix.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`, the
            format specification is 'd' or '.2g' whichever is shorter.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.
        im_kw : dict, default=None
            Dict with keywords passed to `matplotlib.pyplot.imshow` call.

        Returns
        -------
        display : :class:`~skfb.metrics.PAConfusionMatrixDisplay`

        See Also
        --------
        PAConfusionMatrixDisplay.from_estimator

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skfb.estimators import ThresholdFallbackClassifier
        >>> from skfb.metrics import PAConfusionMatrixDisplay
        >>> X = np.array([[0, 0], [4, 4], [1, 1], [3, 3], [2.5, 2], [2., 2.5]])
        >>> y = np.array([0, 1, 0, 1, 0, 1])
        >>> estimator = LogisticRegression(random_state=0)
        >>> rejector = ThresholdFallbackClassifier(estimator, threshold=0.6).fit(X, y)
        >>> y_pred = rejector.predict(X)
        >>> PAConfusionMatrixDisplay.from_predictions(y, y_pred)
        <...>
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_predictions")

        display_labels = display_labels or (False, True)

        cm = predict_accept_confusion_matrix(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            labels=labels,
            normalize=normalize,
        )

        disp = cls(
            confusion_matrix=cm,
            display_labels=display_labels,
            fallback_rate=y_pred.fallback_rate,
        )

        return disp.plot(
            include_values=include_values,
            cmap=cmap,
            ax=ax,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            colorbar=colorbar,
            im_kw=im_kw,
        )


class FQCurveDisplay:
    """Fallback-Quality Curve visualization.

    It is recommend to use
    :func:`~skfb.metrics.FQCurveDisplay.from_estimator` or
    :func:`~skfb.metrics.FQCurveDisplay.from_predictions` to create
    a :class:`~skfb.metrics.FQCurveDisplay`. All parameters are
    stored as attributes.

    Parameters
    ----------
    fallback_rates : array-like
        Rates of rejected samples.
    scores : array-like
        Evaluation scores for every fallback rate.
    fq_auc : float, default=None
        Area under fallback-quality curve.
    estimator_name : str, default=None
        Name of the estimator predicted the scores.
    metric_name : str, default=None
        Name of the scoring method.

    Attributes
    ----------
    line_ : matplotlib Artist
        FQ Curve.
    ax_ : matplotlib Axes
        Axes with FQ Curve.
    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    sfkb.metrics.fallback_quality_curve
    skfb.metrics.fallback_quality_auc_score
    """

    def __init__(
        self,
        *,
        fallback_rates,
        scores,
        fq_auc=None,
        estimator_name=None,
        metric_name=None,
    ):
        self.fallback_rates = fallback_rates
        self.scores = scores
        self.fq_auc = fq_auc
        self.estimator_name = estimator_name
        self.metric_name = metric_name

    def plot(self, ax=None, *, line_kwargs=None, ax_kwargs=None):
        """Plots visualization."""
        check_matplotlib_support(f"{self.__class__.__name__}.plot")

        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt

        if ax is None:
            _, self.ax_ = plt.subplots()
        self.figure_ = self.ax_.figure

        line_kwargs = line_kwargs or {}
        if self.fq_auc is not None and self.estimator_name is not None:
            line_kwargs["label"] = f"{self.estimator_name} (AUC = {self.fq_auc:0.2f})"
        elif self.fq_auc is not None:
            line_kwargs["label"] = f"AUC = {self.fq_auc:0.2f}"
        elif self.estimator_name is not None:
            line_kwargs["label"] = self.estimator_name

        (self.line_,) = self.ax_.plot(self.fallback_rates, self.scores, **line_kwargs)

        xlabel = "Fallback Rate"
        ylabel = self.metric_name or "Prediction Quality"
        ax_kwargs = ax_kwargs or {}
        self.ax_.set(
            xlabel=xlabel,
            xlim=ax_kwargs.get("xlim"),
            ylabel=ylabel,
            ylim=ax_kwargs.get("ylim"),
            aspect="equal",
        )

        if "label" in line_kwargs:
            self.ax_.legend(loc="lower right")

        return self

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        score_func=accuracy_score,
        predict_method="predict",
        min_fallback_rate=0.0,
        max_fallback_rate=0.95,
        raise_warning=True,
        sample_weight=None,
        estimator_name=None,
        metric_name=None,
        ax=None,
        line_kwargs=None,
        ax_kwargs=None,
    ):
        """Plots visualization given an estimator and some data."""
        check_matplotlib_support(f"{cls.__name__}.from_estimator")
        estimator_name = estimator_name or estimator.__class__.__name__

        check_is_fitted(estimator)
        y_score = estimator.predict_proba(X)

        return cls.from_predictions(
            y_true=y,
            y_pred=y_score,
            score_func=score_func,
            predict_method=predict_method,
            raise_warning=raise_warning,
            min_fallback_rate=min_fallback_rate,
            max_fallback_rate=max_fallback_rate,
            sample_weight=sample_weight,
            estimator_name=estimator_name,
            metric_name=metric_name,
            ax=ax,
            line_kwargs=line_kwargs,
            ax_kwargs=ax_kwargs,
        )

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        score_func=accuracy_score,
        predict_method="predict",
        min_fallback_rate=0.0,
        max_fallback_rate=0.95,
        raise_warning=True,
        sample_weight=None,
        estimator_name=None,
        metric_name=None,
        ax=None,
        line_kwargs=None,
        ax_kwargs=None,
    ):
        """Plots visualization given true labels and certainty predictions."""
        check_matplotlib_support(f"{cls.__name__}.from_predictions")
        check_consistent_length(y_true, y_pred, sample_weight)

        fq_curve = fallback_quality_curve(
            y_true,
            y_pred,
            score_func,
            predict_method=predict_method,
            min_fallback_rate=min_fallback_rate,
            max_fallback_rate=max_fallback_rate,
            raise_warning=raise_warning,
        )
        fq_auc = auc(fq_curve.fallback_rates, fq_curve.scores)

        metric_name = metric_name or "Prediction Quality"
        viz = cls(
            fallback_rates=fq_curve.fallback_rates,
            scores=fq_curve.scores,
            fq_auc=fq_auc,
            estimator_name=estimator_name,
            metric_name=metric_name,
        )
        return viz.plot(ax=ax, line_kwargs=line_kwargs, ax_kwargs=ax_kwargs)


class PairedHistogramDisplay:
    """Plots histograms of probabilities of true and false predictions."""

    def __init__(self, score_true, score_false):
        self.score_true = score_true
        self.score_false = score_false

    def plot(self, *, ax=None, cumulative=True):
        """Plots visualization."""
        check_matplotlib_support(f"{self.__class__.__name__}.plot")

        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt

        if ax is None:
            _, self.ax_ = plt.subplots()
        else:
            self.ax_ = ax

        self.ax_.hist(
            self.score_true,
            histtype="step",
            cumulative=cumulative,
            label="Correct",
        )
        self.ax_.hist(
            self.score_false,
            histtype="step",
            cumulative=cumulative,
            label="Incorrect",
        )
        self.ax_.legend()
        self.ax_.grid(visible=True)
        if cumulative:
            self.ax_.set_title("Cumulative distributions of top scores")
        else:
            self.ax_.set_title("Distributions of top scores")
        self.ax_.set_xlabel("Confidence scores")
        self.ax_.set_ylabel("Number of examples")

        return self

    @classmethod
    def from_estimator(cls, estimator, X, y, *, ax=None, cumulative=True):
        """Plots visualization given an estimator and some data."""
        check_matplotlib_support(f"{cls.__name__}.from_estimator")

        check_is_fitted(estimator)
        y_score = estimator.predict_proba(X)

        return cls.from_predictions(
            y_true=y,
            y_score=y_score,
            ax=ax,
            cumulative=cumulative,
        )

    @classmethod
    def from_predictions(cls, y_true, y_score, ax=None, cumulative=True):
        """Plots visualization given true labels and certainty predictions."""
        check_matplotlib_support(f"{cls.__name__}.from_predictions")
        check_consistent_length(y_true, y_score)

        y_pred = y_score.argmax(axis=1)
        true_mask = y_true == y_pred
        y_prob = y_score.max(axis=1)
        score_true = y_prob[true_mask]
        score_false = y_prob[~true_mask]

        viz = cls(score_true, score_false)
        return viz.plot(ax=ax, cumulative=cumulative)
