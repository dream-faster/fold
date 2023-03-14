from typing import List, Optional, Union

import pandas as pd

from ..utils.list import wrap_in_list
from .base import Transformation


class AddLagsY(Transformation):
    properties = Transformation.Properties(mode=Transformation.Properties.Mode.online)

    def __init__(self, lags: Union[List[int], int]) -> None:
        self.lags = wrap_in_list(lags)
        self.max_lag = max(self.lags)
        self.name = f"AddLagsY-{self.lags}"

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.y_in_sample = y.copy()
        self.past_y = y[: -self.max_lag].copy()

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.past_y[y.index[0]] = y.squeeze()
        self.past_y = pd.concat([self.past_y, y], axis="index")[: -self.max_lag].rename(
            "y"
        )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        X = X.copy()
        y = self.y_in_sample if in_sample else self.past_y
        for lag in self.lags:
            X[f"y_lag_{lag}"] = y.shift(lag)
        return X
