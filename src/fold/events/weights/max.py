from typing import Optional, Union

import pandas as pd

from ...base import WeightingStrategy
from ..utils import calculate_rolling_window_size


class WeightByMax(WeightingStrategy):
    def __init__(self, window_size: Optional[Union[int, float]]):
        self.window_size = window_size

    def calculate(self, series: pd.Series) -> pd.Series:
        return calculate_rolling_expanding_abs_max(series, self.window_size)

    def __str__(self) -> str:
        return f"WeightByMax-{self.window_size}"


class WeightByMaxWithLookahead(WeightingStrategy):
    def calculate(self, series: pd.Series) -> pd.Series:
        maximum = series.max()
        sample_weights = series.abs() / maximum
        return sample_weights.fillna(0.0)

    def __str__(self) -> str:
        return "WeightByMaxWithLookahead"


class WeightBySumWithLookahead(WeightingStrategy):
    def calculate(self, series: pd.Series) -> pd.Series:
        sample_weights = series.abs() / series.sum()
        return sample_weights.fillna(0.0)

    def __str__(self) -> str:
        return "WeightBySumWithLookahead"


def calculate_rolling_expanding_abs_max(
    series: pd.Series, window_size: Optional[Union[int, float]]
) -> pd.Series:
    abs_returns = series.abs()
    rolling_or_expanding = (
        abs_returns.expanding()
        if window_size is None
        else abs_returns.rolling(
            calculate_rolling_window_size(window_size, abs_returns), min_periods=0
        )
    )
    maximum = rolling_or_expanding.max()
    sample_weights = series.abs() / maximum
    return sample_weights.fillna(0.0)
