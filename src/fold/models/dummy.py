# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import List, Optional, Union

import pandas as pd

from ..base import Transformation, Tunable, fit_noop
from .base import Model


class DummyClassifier(Model, Tunable):
    """
    A model that predicts a predefined class with predefined probabilities.

    Parameters
    ----------
    predicted_value : float, int
        The class to predict.
    all_classes : List[int]
        All possible classes.
    predicted_probabilities : List[float]
        The probabilities returned.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.models import DummyClassifier
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
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

    ```
    """

    def __init__(
        self,
        predicted_value: Union[float, int],
        all_classes: List[int],
        predicted_probabilities: List[float],
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.predicted_value = predicted_value
        self.all_classes = all_classes
        self.predicted_probabilities = predicted_probabilities
        self.params_to_try = params_to_try
        self.name = name or f"DummyClassifier-{str(self.predicted_value)}"
        self.properties = Transformation.Properties(requires_X=False)

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


class DummyRegressor(Model, Tunable):
    """
    A model that predicts a predefined value.

    Parameters
    ----------
    predicted_value : float
        The value to predict.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.models import DummyRegressor
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = DummyRegressor(0.1)
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                         predictions_DummyRegressor-0.1
    2021-12-31 15:40:00                             0.1
    2021-12-31 15:41:00                             0.1
    2021-12-31 15:42:00                             0.1
    2021-12-31 15:43:00                             0.1
    2021-12-31 15:44:00                             0.1

    ```
    """

    def __init__(
        self,
        predicted_value: float,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.predicted_value = predicted_value
        self.params_to_try = params_to_try
        self.name = name or f"DummyRegressor-{str(self.predicted_value)}"
        self.properties = Transformation.Properties(requires_X=False)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(
            [self.predicted_value] * len(X),
            index=X.index,
        )

    predict_in_sample = predict
    fit = fit_noop
    update = fit
