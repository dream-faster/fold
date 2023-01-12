from enum import Enum

import numpy as np
import pandas as pd

from .base import Model


class BaselineStrategy(Enum):
    sliding_mean = "sliding_mean"
    expanding_mean = "expanding_mean"
    naive = "naive"
    expanding_drift = "drift"
    sliding_drift = "sliding_drift"


class Baseline(Model):

    strategies = BaselineStrategy

    def __init__(self, strategy: BaselineStrategy, window_size: int = 100) -> None:
        self.strategy = strategy
        self.window_size = window_size
        self.name = f"BaselineModel-{strategy.value}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.strategy == BaselineStrategy.sliding_mean:
            return np.array(
                [
                    np.mean(X.values[max(i - self.window_size, 0) : i + 1])
                    for i in range(len(X))
                ]
            )
        elif self.strategy == BaselineStrategy.expanding_mean:
            return np.array([np.mean(X.values[: i + 1]) for i in range(len(X))])
        elif self.strategy == BaselineStrategy.naive:
            return X
        elif self.strategy == BaselineStrategy.sliding_drift:
            return np.array(
                [
                    calculate_drift_predictions(X.values[max(i - self.window_size, 0) : i + 1])
                    for i in range(len(X))
                ]
            )
        elif self.strategy == BaselineStrategy.expanding_drift:
            return np.array([calculate_drift_predictions(X.values[: i + 1]) for i in len(X)])
        else:
            raise ValueError(f"Strategy {self.strategy} not implemented")


def calculate_drift_predictions(y: np.ndarray) -> np.ndarray:
    return y[-1] + np.mean(np.diff(y))
