from typing import Optional

import pandas as pd

from ..base import WeighingStrategy
from ..utils import calculate_rolling_window_size


class WeightByMax(WeighingStrategy):
    def __init__(self, window_size: Optional[int]):
        self.window_size = window_size

    def calculate(self, series: pd.Series) -> pd.Series:
        abs_returns = series.abs()
        rolling_or_expanding = (
            abs_returns.expanding()
            if self.window_size is None
            else abs_returns.rolling(
                calculate_rolling_window_size(self.window_size, abs_returns)
            )
        )
        maximum = rolling_or_expanding.max()
        maximum = maximum.fillna(abs_returns.expanding().max())
        sample_weights = series.abs() / maximum
        return sample_weights.fillna(0.0)


class WeightByMaxWithLookahead(WeighingStrategy):
    def calculate(self, series: pd.Series) -> pd.Series:
        maximum = series.max()
        sample_weights = series.abs() / maximum
        return sample_weights.fillna(0.0)
