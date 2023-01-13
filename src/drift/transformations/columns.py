from typing import List, Optional, Union

import pandas as pd

from .base import Transformation


class SelectColumns(Transformation):
    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = columns if isinstance(columns, List) else [columns]
        self.name = f"SelectColumns-{columns}"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns]


class TransformColumns(Transformation):
    def __init__(
        self, columns: Union[List[str], str], transformation: Transformation
    ) -> None:
        self.columns = columns if isinstance(columns, List) else [columns]
        self.name = f"TransformColumns-{columns}-{transformation.name}"
        self.transformation = transformation

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.transformation.fit(X[self.columns], y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transformation.transform(X[self.columns])
        return X.assign(**transformed.to_dict(orient="series"))
