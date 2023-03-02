from typing import Optional

import pandas as pd

from .base import Transformation


class ToUnivariate(Transformation):

    properties = Transformation.Properties()

    def __init__(self, lag_column: str) -> None:
        self.lag_column = lag_column
        self.name = f"UnivariateWrapper-{lag_column}"

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.lag_column]
