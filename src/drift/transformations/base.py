from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

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


class FeatureSelector(Transformation):
    selected_features: List[str]


class Composite(ABC):
    @abstractmethod
    def get_child_transformations(self) -> Optional[List[Transformation]]:
        raise NotImplementedError

    def set_child_transformations(self, transformations: List[Transformation]) -> None:
        raise NotImplementedError
