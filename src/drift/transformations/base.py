from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd


class Transformation(ABC):

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        raise NotImplementedError

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> Transformation:
        raise NotImplementedError
