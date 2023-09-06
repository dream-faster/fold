import pandas as pd

from ...base import WeightingStrategy
from .max import *  # noqa


class NoWeighting(WeightingStrategy):
    def calculate(self, series: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=series.index)

    def __str__(self) -> str:
        return "NoWeighting"
