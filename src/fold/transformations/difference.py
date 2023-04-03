from typing import Optional

import pandas as pd

from ..base import InvertibleTransformation


class Difference(InvertibleTransformation):
    """
    Performs differencing.
    """

    properties = InvertibleTransformation.Properties(requires_X=False)
    name = "Difference"

    def __init__(self, lag: int = 1) -> None:
        self.lag = lag

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.last_rows_X = X.iloc[-self.lag : None]

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if len(X) >= self.lag:
            self.last_rows_X = X.iloc[-self.lag : None]
        else:
            self.last_rows_X = pd.concat([self.last_rows_X, X], axis="index").iloc[
                -self.lag : None
            ]

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if in_sample:
            return X.diff(self.lag)
        else:
            return (
                pd.concat([self.last_rows_X, X], axis="index")
                .diff(self.lag)
                .iloc[self.lag :]
            )

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        return X.cumsum() + self.last_rows_X.iloc[0].squeeze()
