from typing import Optional

import pandas as pd

from .base import InvertibleTransformation


class Difference(InvertibleTransformation):
    properties = InvertibleTransformation.Properties()

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.y_in_sample = y.copy()
        self.past_y = y[-self.max_lag :].copy()

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if len(y) == 1:
            self.past_y[y.index[0]] = y.squeeze()
        else:
            self.past_y = pd.concat([self.past_y, y], axis="index")
        length_to_keep = len(X) + self.max_lag
        self.past_y = self.past_y[-length_to_keep:]

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        X = X.copy()
        if len(X) > 1:
            y = self.y_in_sample if in_sample else self.past_y
            for lag in self.lags:
                X[f"y_lag_{lag}"] = y.shift(lag)[-len(X) :]
        else:
            y = self.y_in_sample if in_sample else self.past_y
            for lag in self.lags:
                X[f"y_lag_{lag}"] = y[-lag]

        return X
