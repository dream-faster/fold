from typing import List

import pandas as pd

from ...base import LabelingStrategy


class BinarizeSign(LabelingStrategy):
    def label(self, series: pd.Series) -> pd.Series:
        labels = series.copy()
        labels.loc[series >= 0.0] = 1
        labels.loc[series < 0.0] = 0
        return labels

    def get_all_labels(self) -> List[int]:
        return [0, 1]


class NoLabel(LabelingStrategy):
    def label(self, series: pd.Series) -> pd.Series:
        return series

    def get_all_labels(self) -> List[int]:
        return [0, 1]
