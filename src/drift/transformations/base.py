from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import pandas as pd


class Transformation(ABC):

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


class FeatureSelector(Transformation):
    selected_features: List[str]


class Composite(ABC):
    @abstractmethod
    def get_child_transformations(self) -> Transformations:
        raise NotImplementedError

    def before_fit(self, X: pd.DataFrame) -> None:
        pass

    def preprocess_X(self, X: pd.DataFrame, index: int) -> pd.DataFrame:
        return X

    def preprocess_y(self, y: pd.Series) -> pd.Series:
        return y

    @abstractmethod
    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError


Transformations = Union[
    Transformation,
    Composite,
    Callable,
    List[Union[Transformation, Composite, Callable]],
]
