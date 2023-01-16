from typing import Optional

import pandas as pd

from .base import Transformation


class NoTransformation(Transformation):

    name = "No"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
