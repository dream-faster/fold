from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import pandas as pd


class Transformation(ABC):

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @dataclass
    class Properties:
        class ModelType(enum.Enum):
            regressor = "regressor"
            classifier = "classifier"

        requires_past_X: bool  # ignored for now, assumed always True
        model_type: Optional[ModelType] = None


class FeatureSelector(Transformation):
    selected_features: List[str]


class Composite(ABC):
    @abstractmethod
    def get_child_transformations(self) -> Transformations:
        raise NotImplementedError

    @abstractmethod
    def postprocess_result(
        self, results: List[pd.DataFrame], for_inference: bool
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def clone(self, clone_child_transformations: Callable) -> Composite:
        raise NotImplementedError

    def before_fit(self, X: pd.DataFrame) -> None:
        pass

    def preprocess_X(
        self, X: pd.DataFrame, index: int, for_inference: bool
    ) -> pd.DataFrame:
        return X

    def preprocess_y(self, y: pd.Series) -> pd.Series:
        return y


Transformations = Union[
    Transformation,
    Composite,
    Callable,
    List[Union[Transformation, Composite, Callable]],
]
