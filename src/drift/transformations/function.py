from typing import Callable, Optional

import pandas as pd

from .base import Transformation


class FunctionTransformation(Transformation):
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.name = func.__name__

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.func(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X