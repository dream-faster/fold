from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import pandas as pd


class Transformation(ABC):
    @dataclass
    class Properties:
        class ModelType(enum.Enum):
            regressor = "regressor"
            classifier = "classifier"

        requires_past_X: bool  # ignored for now, assumed always True
        model_type: Optional[ModelType] = None

    properties: Properties
    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class FeatureSelector(Transformation):
    selected_features: List[str]


class Composite(ABC):
    @dataclass
    class Properties:
        primary_requires_predictions: bool = False
        primary_only_single_pipeline: bool = False
        secondary_requires_predictions: bool = False
        secondary_only_single_pipeline: bool = False

    properties: Properties

    @abstractmethod
    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        raise NotImplementedError

    def get_child_transformations_secondary(
        self,
    ) -> Optional[TransformationsAlwaysList]:
        return None

    @abstractmethod
    def postprocess_result_primary(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError

    def postprocess_result_secondary(
        self,
        primary_results: List[pd.DataFrame],
        secondary_results: List[pd.DataFrame],
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def clone(self, clone_child_transformations: Callable) -> Composite:
        raise NotImplementedError

    def before_fit(self, X: pd.DataFrame) -> None:
        pass

    def preprocess_X_primary(self, X: pd.DataFrame, index: int) -> pd.DataFrame:
        return X

    def preprocess_X_secondary(
        self, X: pd.DataFrame, results_primary: List[pd.DataFrame], index: int
    ) -> pd.DataFrame:
        return X

    def preprocess_y_primary(self, y: pd.Series) -> pd.Series:
        return y

    def preprocess_y_secondary(
        self, y: pd.Series, results_primary: List[pd.DataFrame]
    ) -> pd.Series:
        return y


Transformations = Union[
    Transformation,
    Composite,
    Callable,
    List[Union[Transformation, Composite, Callable]],
]

TransformationsAlwaysList = List[Union[Transformation, Composite, Callable]]
