from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TypeVar

import pandas as pd

from ..transformations.base import TransformationsAlwaysList

T = TypeVar("T", Optional[pd.Series], pd.Series)


class Composite(ABC):
    """
    A Composite transformation is a transformation that contains other transformations.
    """

    @dataclass
    class Properties:
        primary_requires_predictions: bool = (
            False  # Primary transformations need output from a model
        )
        primary_only_single_pipeline: bool = False  # Primary transformations should contain only a single pipeline, not multiple.
        secondary_requires_predictions: bool = (
            False  # Secondary transformations need output from a model
        )
        secondary_only_single_pipeline: bool = False  # Secondary transformations should contain only a single pipeline, not multiple.

    properties: Properties

    @abstractmethod
    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        raise NotImplementedError

    def get_child_transformations_secondary(
        self,
    ) -> Optional[TransformationsAlwaysList]:
        return None

    @abstractmethod
    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        raise NotImplementedError

    def postprocess_result_secondary(
        self,
        primary_results: List[pd.DataFrame],
        secondary_results: List[pd.DataFrame],
        y: Optional[pd.Series],
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def clone(self, clone_child_transformations: Callable) -> "Composite":
        raise NotImplementedError

    def before_fit(self, X: pd.DataFrame) -> None:
        pass

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        return X, y

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        results_primary: List[pd.DataFrame],
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T]:
        return X, y
