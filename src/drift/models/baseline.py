from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .base import Model


class BaselineStrategy(Enum):
    sliding_mean = "sliding_mean"
    expanding_mean = "expanding_mean"
    seasonal_mean = "seasonal_mean"
    naive = "naive"
    seasonal_naive = "seasonal_naive"
    expanding_drift = "drift"
    sliding_drift = "sliding_drift"


class Baseline(Model):

    strategies = BaselineStrategy

    def __init__(
        self,
        strategy: BaselineStrategy,
        window_size: int = 100,
        seasonal_length: Optional[int] = None,
    ) -> None:
        self.strategy = strategy
        self.window_size = window_size
        self.seasonal_length = seasonal_length
        self.name = f"BaselineModel-{strategy.value}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.fitted_X = X

    def predict(self, X: pd.DataFrame) -> pd.Series:
        extended_X = pd.concat([self.fitted_X, X], axis=0)

        def wrap_into_series(x: np.ndarray) -> pd.Series:
            return pd.Series(x, index=X.index)

        if self.strategy == BaselineStrategy.sliding_mean:
            return wrap_into_series(
                [
                    np.mean(X.values[max(i - self.window_size, 0) : i + 1])
                    for i in range(len(X))
                ]
            )
        elif self.strategy == BaselineStrategy.expanding_mean:
            return wrap_into_series([np.mean(X.values[: i + 1]) for i in range(len(X))])
        elif self.strategy == BaselineStrategy.naive:
            return X
        elif self.strategy == BaselineStrategy.sliding_drift:
            return wrap_into_series(
                [
                    calculate_drift_predictions(
                        X.values[max(i - self.window_size, 0) : i + 1]
                    )
                    for i in range(len(X))
                ]
            )
        elif self.strategy == BaselineStrategy.expanding_drift:
            return wrap_into_series(
                [calculate_drift_predictions(X.values[: i + 1]) for i in len(X)]
            )
        elif self.strategy == BaselineStrategy.seasonal_naive:
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
        elif self.strategy == BaselineStrategy.seasonal_mean:
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


def calculate_drift_predictions(y: np.ndarray) -> np.ndarray:
    return y[-1] + np.mean(np.diff(y))
