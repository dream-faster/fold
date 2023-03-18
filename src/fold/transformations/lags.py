from typing import List, Tuple, Union

import pandas as pd

from ..utils.list import flatten, wrap_in_list
from .base import Transformation, fit_noop


class AddLagsY(Transformation):
    def __init__(self, lags: Union[List[int], int]) -> None:
        self.lags = wrap_in_list(lags)
        self.name = f"AddLagsY-{self.lags}"
        self.properties = Transformation.Properties(
            mode=Transformation.Properties.Mode.online, memory=max(self.lags)
        )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        X = X.copy()
        if in_sample:
            for lag in self.lags:
                X[f"y_lag_{lag}"] = self._state.memory_y.shift(lag)[-len(X) :]
            return X
        else:
            past_y = self._state.memory_y.reindex(X.index)
            for lag in self.lags:
                X[f"y_lag_{lag}"] = past_y.shift(lag)[-len(X) :]
            return X

    fit = fit_noop
    update = fit_noop


class AddLagsX(Transformation):
    ColumnAndLag = Tuple[str, Union[int, List[int]]]

    def __init__(
        self, columns_and_lags: Union[List[ColumnAndLag], ColumnAndLag]
    ) -> None:
        self.columns_and_lags = wrap_in_list(columns_and_lags)
        self.name = f"AddLagsX-{self.columns_and_lags}"
        self.properties = Transformation.Properties(
            memory=max(flatten([l for _, l in self.columns_and_lags]))
        )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        X = X.copy()
        for column, lags in self.columns_and_lags:
            lags = wrap_in_list(lags)
            for lag in lags:
                X[f"{column}_lag_{lag}"] = X[column].shift(lag)[-len(X) :]
        return X

    fit = fit_noop
    update = fit_noop
