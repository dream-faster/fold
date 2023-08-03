# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from ..base import Tunable, fit_noop
from .base import TimeSeriesModel


class Naive(TimeSeriesModel):
    """
    A univariate model that predicts the last target value.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.models import Naive
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = Naive()
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> pd.concat([preds, y[preds.index]], axis=1).head()
                         predictions_Naive    sine
    2021-12-31 15:40:00            -0.0000  0.0126
    2021-12-31 15:41:00             0.0126  0.0251
    2021-12-31 15:42:00             0.0251  0.0377
    2021-12-31 15:43:00             0.0377  0.0502
    2021-12-31 15:44:00             0.0502  0.0628

    ```
    """

    name = "Naive"
    properties = TimeSeriesModel.Properties(
        requires_X=False,
        mode=TimeSeriesModel.Properties.Mode.online,
        memory_size=1,
        _internal_supports_minibatch_backtesting=True,
    )

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(past_y.iloc[-1].squeeze(), index=X.index[-1:None]).fillna(0.0)

    def predict_in_sample(
        self, X: pd.DataFrame, lagged_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return lagged_y.fillna(0.0)

    fit = fit_noop
    update = fit


class NaiveSeasonal(TimeSeriesModel, Tunable):
    """
    A model that predicts the last value seen in the same season.
    """

    name = "NaiveSeasonal"

    def __init__(
        self, seasonal_length: int, params_to_try: Optional[dict] = None
    ) -> None:
        assert seasonal_length > 1, "seasonal_length must be greater than 1"
        self.seasonal_length = seasonal_length
        self.properties = TimeSeriesModel.Properties(
            requires_X=False,
            mode=TimeSeriesModel.Properties.Mode.online,
            memory_size=seasonal_length,
            _internal_supports_minibatch_backtesting=True,
        )
        self.params_to_try = params_to_try

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            past_y.iloc[-self.seasonal_length].squeeze(),
            index=X.index[-1:None],
        ).fillna(0.0)

    def predict_in_sample(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return past_y.shift(self.seasonal_length - 1).fillna(0.0)

    fit = fit_noop
    update = fit

    def get_params(self) -> dict:
        return {"seasonal_length": self.seasonal_length}


class MovingAverage(TimeSeriesModel, Tunable):
    """
    A model that predicts the mean of the last values seen.
    """

    name = "MovingAverage"

    def __init__(self, window_size: int, params_to_try: Optional[dict] = None) -> None:
        self.window_size = window_size
        self.properties = TimeSeriesModel.Properties(
            requires_X=False,
            mode=TimeSeriesModel.Properties.Mode.online,
            memory_size=window_size,
            _internal_supports_minibatch_backtesting=True,
        )
        self.params_to_try = params_to_try

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            past_y.rolling(self.window_size, min_periods=0).mean()[-1],
            index=X.index[-1:None],
        ).fillna(0.0)

    def predict_in_sample(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return past_y.rolling(self.window_size, min_periods=0).mean().fillna(0.0)

    fit = fit_noop
    update = fit

    def get_params(self) -> dict:
        return {"window_size": self.window_size}


class ExponentiallyWeightedMovingAverage(TimeSeriesModel, Tunable):
    """
    A model that predicts the exponentially weighed mean of the last values seen.
    """

    name = "ExponentiallyWeightedMovingAverage"

    def __init__(self, window_size: int, params_to_try: Optional[dict] = None) -> None:
        self.window_size = window_size
        self.properties = TimeSeriesModel.Properties(
            requires_X=False,
            mode=TimeSeriesModel.Properties.Mode.online,
            memory_size=self.window_size * 4,
            _internal_supports_minibatch_backtesting=True,
        )
        self.params_to_try = params_to_try

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            past_y.ewm(alpha=1 / self.window_size, adjust=True, min_periods=0).mean()[
                -1
            ],
            index=X.index[-1:None],
        ).fillna(0.0)

    def predict_in_sample(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return (
            past_y.ewm(alpha=1 / self.window_size, adjust=True, min_periods=0)
            .mean()
            .fillna(0.0)
        )

    fit = fit_noop
    update = fit

    def get_params(self) -> dict:
        return {"window_size": self.window_size}
