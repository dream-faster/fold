from typing import List, Union

import pandas as pd

from ..base import Transformation, fit_noop
from .base import Model


class DummyClassifier(Model):
    """
    A model that predicts a predefined class with predefined probabilities.

    Parameters
    ----------
    predicted_value : Union[float, int]
        The class to predict.
    all_classes : List[int]
        All possible classes.
    predicted_probabilities : List[float]
        The probabilities returned.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.models import DummyClassifier
        >>> from fold.utils.tests import generate_sine_wave_data
        >>> X, y  = generate_sine_wave_data()
        >>> splitter = SlidingWindowSplitter(initial_train_window=0.5, step=0.2)
        >>> pipeline = DummyClassifier(1, [0, 1], [0.5, 0.5])
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
        >>> preds.head()
                             predictions_DummyClassifier  ...  probabilities_DummyClassifier_1
        2021-12-31 15:40:00                            1  ...                              0.5
        2021-12-31 15:41:00                            1  ...                              0.5
        2021-12-31 15:42:00                            1  ...                              0.5
        2021-12-31 15:43:00                            1  ...                              0.5
        2021-12-31 15:44:00                            1  ...                              0.5
        <BLANKLINE>
        [5 rows x 3 columns]
    """

    properties = Transformation.Properties(requires_X=False)
    name = "DummyClassifier"

    def __init__(
        self,
        predicted_value: Union[float, int],
        all_classes: List[int],
        predicted_probabilities: List[float],
    ) -> None:
        self.predicted_value = predicted_value
        self.all_classes = all_classes
        self.predicted_probabilities = predicted_probabilities

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        predictions = pd.Series(
            [self.predicted_value] * len(X),
            index=X.index,
            name="predictions_DummyClassifier",
        )
        probabilities = [
            pd.Series(
                [prob] * len(X),
                index=X.index,
                name=f"probabilities_DummyClassifier_{associated_class}",
            )
            for associated_class, prob in zip(
                self.all_classes, self.predicted_probabilities
            )
        ]

        return pd.concat([predictions] + probabilities, axis="columns")

    predict_in_sample = predict
    fit = fit_noop
    update = fit


class DummyRegressor(Model):
    """
    A model that predicts a predefined value.

    Parameters
    ----------
    predicted_value : float
        The value to predict.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.models import DummyRegressor
        >>> from fold.utils.tests import generate_sine_wave_data
        >>> X, y  = generate_sine_wave_data()
        >>> splitter = SlidingWindowSplitter(initial_train_window=0.5, step=0.2)
        >>> pipeline = DummyRegressor(0.1)
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
        >>> preds.head()
                             predictions_DummyRegressor
        2021-12-31 15:40:00                         0.1
        2021-12-31 15:41:00                         0.1
        2021-12-31 15:42:00                         0.1
        2021-12-31 15:43:00                         0.1
        2021-12-31 15:44:00                         0.1
    """

    properties = Transformation.Properties(requires_X=False)
    name = "DummyRegressor"

    def __init__(self, predicted_value: float) -> None:
        self.predicted_value = predicted_value

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(
            [self.predicted_value] * len(X),
            index=X.index,
        )

    predict_in_sample = predict
    fit = fit_noop
    update = fit
