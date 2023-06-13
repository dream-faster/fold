from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TypeVar, Union

import pandas as pd

from fold.utils.dataframe import ResolutionStrategy, concat_on_columns_with_duplicates
from fold.utils.list import filter_none

from ..splitters import SingleWindowSplitter
from ..utils.introspection import get_initialization_parameters

T = TypeVar("T", Optional[pd.Series], pd.Series)
X = pd.DataFrame


class Block(ABC):
    id: str
    name: str

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.id = str(uuid.uuid4())
        return instance


class Composite(Block, ABC):
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
    def get_children_primary(self) -> Pipelines:
        raise NotImplementedError

    def get_children_secondary(
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
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        return X, y, artifact

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        artifact: Artifact,
        results_primary: List[pd.DataFrame],
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        return X, y, artifact

    def postprocess_artifacts_primary(
        self,
        artifacts: List[Artifact],
        results: List[pd.DataFrame],
        fit: bool,
    ) -> pd.DataFrame:
        return pd.concat(artifacts, axis="columns")

    def postprocess_artifacts_secondary(
        self, primary_artifacts: pd.DataFrame, secondary_artifacts: List[Artifact]
    ) -> pd.DataFrame:
        return pd.concat(
            [
                primary_artifacts,
                pd.concat(secondary_artifacts, axis="columns"),
            ],
            axis="columns",
        )


class Optimizer(Block, ABC):
    splitter: SingleWindowSplitter

    @abstractmethod
    def get_candidates(self) -> List[Pipeline]:
        """
        Called iteratively, until an array with a length of zero is returned.
        Then the loop finishes the candidate evaluation process.
        """
        raise NotImplementedError

    @abstractmethod
    def get_optimized_pipeline(self) -> Optional[Pipeline]:
        raise NotImplementedError

    @abstractmethod
    def process_candidate_results(
        self,
        results: List[pd.DataFrame],
        y: pd.Series,
        artifacts: List[pd.DataFrame],
    ) -> Optional[Artifact]:
        raise NotImplementedError

    @abstractmethod
    def clone(self, clone_children: Callable) -> Optimizer:
        raise NotImplementedError


class Transformation(Block, ABC):
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


class Tunable(Block, ABC):
    params_to_try: Optional[dict]

    def get_params(self) -> dict:
        """
        The default implementation assumes that:
        1. All init parameters are stored on the object as property (with the same name/key).
        2. There are no modifications/conversions of the init parameters that'd prevent them from being used again (reconstructing the object from them).
        """
        return get_initialization_parameters(self)

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

TrainedPipeline = Pipeline
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


class EventDataFrame(pd.DataFrame):
    start: pd.Series
    end: pd.Series
    label: pd.Series
    raw: pd.Series
    sample_weights: pd.Series
    test_sample_weights: pd.Series

    @classmethod
    def from_data(
        cls,
        start: pd.DatetimeIndex,
        end: pd.DatetimeIndex,
        label: pd.Series,
        raw: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        test_sample_weights: Optional[pd.Series] = None,
    ) -> EventDataFrame:
        cls(
            data={
                "start": start,
                "end": end,
                "label": label,
                "raw": raw,
                "sample_weights": pd.Series(1.0, index=start)
                if sample_weights is None
                else sample_weights,
                "test_sample_weights": pd.Series(1.0, index=start)
                if test_sample_weights is None
                else test_sample_weights,
            }
        )

    @staticmethod
    def get_columns() -> List[str]:
        return ["start", "end", "label", "raw", "sample_weights", "test_sample_weights"]


class Artifact(pd.DataFrame):
    @staticmethod
    def empty(index: pd.Index) -> Artifact:
        return pd.DataFrame(index=index)  # type: ignore

    @staticmethod
    def get_sample_weights(artifact: Artifact) -> Optional[pd.Series]:
        if "sample_weights" not in artifact.columns:
            return None
        return artifact["sample_weights"]

    @staticmethod
    def get_test_sample_weights(artifact: Artifact) -> Optional[pd.Series]:
        if "test_sample_weights" not in artifact.columns:
            if "sample_weights" in artifact.columns:
                return artifact["sample_weights"]
            else:
                return None
        return artifact["test_sample_weights"]

    @staticmethod
    def get_label(artifact: Artifact) -> Optional[pd.Series]:
        if "label" not in artifact.columns:
            return None
        return artifact["label"]

    @staticmethod
    def get_events(artifact: Artifact) -> Optional[EventDataFrame]:
        if "start" not in artifact.columns:
            return None
        return EventDataFrame.from_data(
            start=artifact.start,
            end=artifact.index,
            label=artifact.label,
            raw=artifact.raw,
            sample_weights=artifact.sample_weights,
            test_sample_weights=artifact.test_sample_weights,
        )

    @staticmethod
    def from_events_sample_weights(
        index: pd.Index,
        events: Optional[EventDataFrame],
        sample_weights: Optional[pd.Series],
    ) -> Artifact:
        result = concat_on_columns_with_duplicates(
            filter_none([events, sample_weights]), strategy=ResolutionStrategy.first
        )
        if result.empty:
            return Artifact.empty(index)  # type: ignore
        else:
            return result  # type: ignore
