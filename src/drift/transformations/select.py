from typing import List, Optional, Union

import pandas as pd

from .base import Transformation


class SelectColumn(Transformation):
    def __init__(self, column: Union[List[str], str]) -> None:
        self.column = column
        self.name = f"SelectColumn-{column}"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.column if self.column is List else [self.column]]
