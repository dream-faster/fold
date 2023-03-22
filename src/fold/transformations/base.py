from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Union

if TYPE_CHECKING:
    from ..composites.base import Composite

import pandas as pd
from sklearn.base import BaseEstimator


class Transformation(ABC):
    @dataclass
    class Properties:
        class ModelType(enum.Enum):
            regressor = "regressor"
            classifier = "classifier"

        class Mode(enum.Enum):
            minibatch = "minibatch"
            online = "online"

        mode: Mode = Mode.minibatch
        memory_size: Optional[
            int
        ] = None  # if not `None`, will inject past window with size of `memory` to update() & transformation(). if `0`, it'll remember all data. during the in_sample period, it'll contain all data.
        model_type: Optional[ModelType] = None

    @dataclass
    class State:
        memory_X: pd.DataFrame
        memory_y: Optional[pd.Series]

    properties: Properties
    name: str
    _state: Optional[State] = None

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        """
        Called once, with on initial training window.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        """
        Subsequent calls to update the model, on each fold.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        raise NotImplementedError


class InvertibleTransformation(Transformation, ABC):
    @abstractmethod
    def inverse_transform(self, X: pd.Series) -> pd.Series:
        raise NotImplementedError


class FeatureSelector(Transformation):
    selected_features: List[str]


Transformations = Union[
    Transformation,
    "Composite",
    List[Union[Transformation, "Composite"]],
]
DeployableTransformations = Transformations

Pipeline = List[Union[Transformation, "Composite"]]
Pipelines = List[Pipeline]

BlockOrWrappable = Union[Transformation, "Composite", Callable, BaseEstimator]
BlocksOrWrappable = Union[BlockOrWrappable, List[BlockOrWrappable]]


def fit_noop(
    self,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series] = None,
) -> None:
    pass
