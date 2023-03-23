from __future__ import annotations

from typing import Union

import pandas as pd

from ..transformations.base import Transformation, fit_noop
from .base import Model


class Naive(Model):
    """
    A model that predicts the last value seen.
    """

    name = "Naive"
    properties = Model.Properties(
        mode=Transformation.Properties.Mode.online, memory_size=1
    )

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            self._state.memory_y.iloc[-1].squeeze(), index=X.index[-1:None]
        )

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self._state.memory_y.shift(1)

    fit = fit_noop
    update = fit


class NaiveSeasonal(Model):
    """
    A model that predicts the last value seen in the same season.
    """

    name = "NaiveSeasonal"

    def __init__(self, seasonal_length: int) -> None:
        assert seasonal_length > 1, "seasonal_length must be greater than 1"
        self.seasonal_length = seasonal_length
        self.properties = Model.Properties(
            mode=Transformation.Properties.Mode.online,
            memory_size=seasonal_length,
            _internal_supports_minibatch_backtesting=True,
        )

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            self._state.memory_y.iloc[-self.seasonal_length].squeeze(),
            index=X.index[-1:None],
        )

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self._state.memory_y.shift(self.seasonal_length)

    fit = fit_noop
    update = fit


class RollingMean(Model):
    """
    A model that predicts the mean of the last values seen.
    """

    name = "RollingMean"

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.properties = Model.Properties(
            mode=Transformation.Properties.Mode.online, memory_size=window_size
        )

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            self._state.memory_y[-self.window_size :].mean(), index=X.index[-1:None]
        )

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self._state.memory_y.shift(1).rolling(self.window_size).mean()

    fit = fit_noop
    update = fit
