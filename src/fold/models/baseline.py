from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..transformations.base import Transformation
from .base import Model


class BaselineRegressorDeprecated(Model):
    class Strategy(Enum):
        sliding_mean = "sliding_mean"
        expanding_mean = "expanding_mean"
        seasonal_mean = "seasonal_mean"
        naive = "naive"
        seasonal_naive = "seasonal_naive"
        expanding_fold = "expanding_fold"
        sliding_fold = "sliding_fold"

        @staticmethod
        def from_str(
            value: Union[str, BaselineRegressorDeprecated.Strategy]
        ) -> BaselineRegressorDeprecated.Strategy:
            if isinstance(value, BaselineRegressorDeprecated.Strategy):
                return value
            for strategy in BaselineRegressorDeprecated.Strategy:
                if strategy.value == value:
                    return strategy
            else:
                raise ValueError(f"Unknown Strategy: {value}")

    properties = Transformation.Properties()

    def __init__(
        self,
        strategy: Union[BaselineRegressorDeprecated.Strategy, str],
        window_size: int = 100,
        seasonal_length: Optional[int] = None,
    ) -> None:
        self.strategy = BaselineRegressorDeprecated.Strategy.from_str(strategy)
        self.window_size = window_size
        self.seasonal_length = seasonal_length
        self.name = f"BaselineModel-{self.strategy.value}"

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.fitted_X = X

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        def wrap_into_series(x: np.ndarray) -> pd.Series:
            return pd.Series(x, index=X.index)

        if self.strategy == BaselineRegressorDeprecated.Strategy.sliding_mean:
            return wrap_into_series(
                [
                    np.mean(X.values[max(i - self.window_size, 0) : i + 1])
                    for i in range(len(X))
                ]
            )
        elif self.strategy == BaselineRegressorDeprecated.Strategy.expanding_mean:
            return wrap_into_series([np.mean(X.values[: i + 1]) for i in range(len(X))])
        elif self.strategy == BaselineRegressorDeprecated.Strategy.naive:
            return X
        elif self.strategy == BaselineRegressorDeprecated.Strategy.sliding_fold:
            return wrap_into_series(
                [
                    calculate_fold_predictions(
                        X.values[max(i - self.window_size, 0) : i + 1]
                    )
                    for i in range(len(X))
                ]
            )
        elif self.strategy == BaselineRegressorDeprecated.Strategy.expanding_fold:
            return wrap_into_series(
                [calculate_fold_predictions(X.values[: i + 1]) for i in len(X)]
            )
        elif self.strategy == BaselineRegressorDeprecated.Strategy.seasonal_naive:
            if self.seasonal_length is None:
                raise ValueError(
                    "Seasonal length must be specified for seasonal naive strategy"
                )
            seasonally_shifted = [
                X.values[i - self.seasonal_length] for i in range(len(X))
            ]
            # but prevent lookahead bias
            seasonally_shifted[: self.seasonal_length] = X.values[
                : self.seasonal_length
            ]
            return wrap_into_series(seasonally_shifted)
        elif self.strategy == BaselineRegressorDeprecated.Strategy.seasonal_mean:
            if self.seasonal_length is None:
                raise ValueError(
                    "Seasonal length must be specified for seasonal naive strategy"
                )
            seasonal_means = [
                np.nanmean(X.shift(season).values[:: self.seasonal_length])
                for season in range(self.seasonal_length)
            ]

            return wrap_into_series(
                [seasonal_means[i % self.seasonal_length] for i in range(len(X))]
            )
        else:
            raise ValueError(f"Strategy {self.strategy} not implemented")

    predict_in_sample = predict


class BaselineNaive(Model):
    name = "BaselineNaive"
    properties = Model.Properties(requires_continuous_updates=True)
    past_y = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.past_y = y[-1]

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(self.past_y, index=X.index)

    predict_in_sample = predict


class BaselineNaiveSeasonal(Model):
    name = "BaselineNaiveSeasonal"
    properties = Model.Properties(requires_continuous_updates=True)
    current_season = 0

    def __init__(self, seasonal_length: int) -> None:
        self.seasonal_length = seasonal_length
        self.past_ys = [None] * seasonal_length

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.past_ys[self.current_season % self.seasonal_length] = y[-1]
        self.current_season += 1

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        index = self.current_season % self.seasonal_length
        value = self.past_ys[index] if self.past_ys[index] is not None else 0.0
        return pd.Series([value], index=X.index)

    predict_in_sample = predict


# class BaselineMean(Model):

#     name = "BaselineMean"
#     properties = Transformation.Properties(requires_continuous_updates=True)
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
