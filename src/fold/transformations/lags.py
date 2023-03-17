from typing import List, Union

import pandas as pd

from ..utils.list import wrap_in_list
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
        for lag in self.lags:
            X[f"y_lag_{lag}"] = self._state.memory_y.shift(lag)[-len(X) :]
        return X

    fit = fit_noop
    update = fit_noop
