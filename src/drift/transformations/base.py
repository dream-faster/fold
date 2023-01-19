from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Self, Union

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
    def clone(self) -> Self:
        raise NotImplementedError


class FeatureSelector(Transformation):
    selected_features: List[str]


class Composite(ABC):
    @abstractmethod
    def get_child_transformations(self) -> List[Transformations]:
        raise NotImplementedError

    def set_child_transformations(self, transformations: List[Transformations]) -> None:
        raise NotImplementedError

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError


Transformations = Union[
    Transformation,
    Composite,
    Callable,
    List[Union[Transformation, Composite, Callable]],
]
