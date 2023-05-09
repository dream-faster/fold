# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import pandas as pd

from fold.splitters import SingleWindowSplitter

T = TypeVar("T", Optional[pd.Series], pd.Series)
V = TypeVar("V", Optional[pd.Series], pd.Series)
X = pd.DataFrame
Artifact = pd.DataFrame


class Composite(ABC):
    """
    A Composite contains other transformations.
    """

    @dataclass
    class Properties:
        primary_requires_predictions: bool = (
            False  # Primary pipeline need output from a model
        )
        primary_only_single_pipeline: bool = (
            False  # There should be a single primary pipeline.
        )
        secondary_requires_predictions: bool = (
            False  # Secondary pipeline need output from a model
        )
        secondary_only_single_pipeline: bool = (
            False  # There should be a single secondary pipeline.
        )

    properties: Properties

    @abstractmethod
    def get_child_transformations_primary(self) -> Pipelines:
        raise NotImplementedError

    def get_child_transformations_secondary(
        self,
    ) -> Optional[Pipelines]:
        return None

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        raise NotImplementedError

    def postprocess_result_secondary(
        self,
        primary_results: List[pd.DataFrame],
        secondary_results: List[pd.DataFrame],
        y: Optional[pd.Series],
        in_sample: bool,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def clone(self, clone_children: Callable) -> Composite:
        raise NotImplementedError

    def before_fit(self, X: pd.DataFrame) -> None:
        pass

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, sample_weights: V, fit: bool
    ) -> Tuple[pd.DataFrame, T, V]:
        return X, y, sample_weights

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        results_primary: List[pd.DataFrame],
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T]:
        return X, y

    def postprocess_artifacts_primary(self, artifacts: List[Artifact]) -> pd.DataFrame:
        return pd.concat(artifacts, axis="columns").add_prefix("primary_")

    def postprocess_artifacts_secondary(
        self, primary_artifacts: pd.DataFrame, secondary_artifacts: List[Artifact]
    ) -> pd.DataFrame:
        return pd.concat(
            [
                primary_artifacts,
                pd.concat(secondary_artifacts, axis="columns").add_prefix("secondary_"),
            ],
            axis="columns",
        )


class Optimizer(ABC):
    splitter: SingleWindowSplitter

    @abstractmethod
    def get_candidates(self) -> Iterable["Pipeline"]:
        raise NotImplementedError

    @abstractmethod
    def get_optimized_pipeline(self) -> Optional["Pipeline"]:
        raise NotImplementedError

    @abstractmethod
    def process_candidate_results(
        self, results: List[pd.DataFrame], y: pd.Series
    ) -> Optional[Artifact]:
        raise NotImplementedError

    @abstractmethod
    def clone(self, clone_children: Callable) -> Optimizer:
        raise NotImplementedError


class Transformation(ABC):
    """
    A transformation is a single step in a pipeline.
    """

    @dataclass
    class Properties:
        class ModelType(enum.Enum):
            regressor = "regressor"
            classifier = "classifier"

        class Mode(enum.Enum):
            minibatch = "minibatch"
            online = "online"

        requires_X: bool
        mode: Mode = Mode.minibatch
        memory_size: Optional[
            int
        ] = None  # if not `None`, will inject past window with size of `memory` to update() & transformation(). if `0`, it'll remember all data. during the in_sample period, it'll contain all data.
        model_type: Optional[ModelType] = None
        _internal_supports_minibatch_backtesting: bool = False  # internal, during backtesting, calls predict_in_sample() instead of predict()

    @dataclass
    class State:
        memory_X: pd.DataFrame
        memory_y: pd.Series
        memory_sample_weights: Optional[pd.Series]

    properties: Properties
    name: str
    _state: Optional[State] = None

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        """
        Called once, with on initial training window.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        """
        Subsequent calls to update the model.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        raise NotImplementedError


class InvertibleTransformation(Transformation, ABC):
    @abstractmethod
    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        raise NotImplementedError


class Tunable(ABC):
    params_to_try: Optional[dict]

    def __init__(self) -> None:
        self.id = str(uuid.uuid4())

    @abstractmethod
    def get_params(self) -> dict:
        raise NotImplementedError

    def get_params_to_try(self) -> Optional[dict]:
        return self.params_to_try

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        """
        The default implementation only works for Transformations, when parameters and the init parameters match 100%.
        """
        return self.__class__(**parameters)


class FeatureSelector(Transformation):
    selected_features: List[str]


Transformations = Union[
    Transformation,
    "Composite",
    List[Union[Transformation, "Composite"]],
]

Pipeline = Union[
    Union[Transformation, "Composite"], List[Union[Transformation, "Composite"]]
]
"""A list of `fold` objects that are executed sequentially. Or a single object."""
Pipelines = List[Pipeline]
"""Multiple, independent `Pipeline`s."""

TrainedPipelines = List[pd.Series]
"""A list of trained `Pipeline`s, to be used for backtesting."""
OutOfSamplePredictions = pd.DataFrame
"""The backtest's resulting out-of-sample predictions."""
DeployablePipeline = Pipeline
"""A trained `Pipeline`, to be used for deployment."""


def fit_noop(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series] = None,
) -> None:
    pass


class SingleFunctionTransformation(Transformation):
    properties = Transformation.Properties(requires_X=True)

    def get_function(self) -> Callable:
        raise NotImplementedError

    fit = fit_noop
    update = fit_noop

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        return pd.concat([X, self.get_function()(X)], axis="columns"), None
