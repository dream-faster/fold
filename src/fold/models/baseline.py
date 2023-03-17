from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from ..transformations.base import Transformation, fit_noop
from .base import Model


class BaselineNaive(Model):
    name = "BaselineNaive"
    properties = Model.Properties(mode=Transformation.Properties.Mode.online, memory=1)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            self._state.memory_y.iloc[-1].squeeze(), index=X.index[-1:None]
        )

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self._state.memory_y.shift(1)

    fit = fit_noop
    update = fit


class BaselineNaiveSeasonal(Model):
    name = "BaselineNaiveSeasonal"
    properties = Model.Properties(mode=Transformation.Properties.Mode.online)
    current_season = 0
    insample_y = None

    def __init__(self, seasonal_length: int) -> None:
        self.seasonal_length = seasonal_length
        self.past_ys = [None] * seasonal_length

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.insample_y = y.shift(self.seasonal_length)
        self.current_season = len(X) % self.seasonal_length
        last_batch = y[-self.seasonal_length :]
        for value in last_batch:
            self.update(None, pd.Series([value]))

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.past_ys[self.current_season % self.seasonal_length] = y.squeeze()
        self.current_season += 1

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        index = self.current_season % self.seasonal_length
        value = self.past_ys[index] if self.past_ys[index] is not None else 0.0
        return pd.Series([value], index=X.index)

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self.insample_y


# class BaselineMean(Model):

#     name = "BaselineMean"
#     properties = Transformation.Properties(mode=Transformation.Properties.Mode.online)
#     length = 0
#     rolling_mean = None

#     def __init__(self, window_size: Optional[int] = None) -> None:
#         self.name = "BaselineMean"
#         self.window_size = window_size

#     def fit(
#         self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
#     ) -> None:
#         if self.rolling_mean is None:
#             self.rolling_mean = (
#                 y.mean() if sample_weights is None else (y * sample_weights).mean()
#             )
#             self.length = len(y)
#         else:
#             self.rolling_mean = (
#                 self.rolling_mean * self.length + y.mean() * len(y)
#             ) / (self.length + len(y))
#             self.length += len(y)

#     def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
#         if self.rolling_mean is None:
#             return pd.Series(np.zeros(len(X)), index=X.index)
#         return pd.Series([self.rolling_mean], index=X.index)


def calculate_fold_predictions(y: np.ndarray) -> np.ndarray:
    return y[-1] + np.mean(np.diff(y))
