import pandas as pd

from ..base import WeighingStrategy
from .max import *  # noqa


class NoWeighing(WeighingStrategy):
    def calculate(self, series: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=series.index)
