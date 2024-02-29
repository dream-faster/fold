from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self, TypeVar

import pandas as pd
from finml_utils.dataframes import concat_on_columns
from finml_utils.enums import ParsableEnum
from finml_utils.introspection import get_initialization_parameters

from ..splitters import Splitter
from ..utils.dataframe import ResolutionStrategy, concat_on_columns_with_duplicates

T = TypeVar("T", pd.Series | None, pd.Series)
X = pd.DataFrame


class Block(ABC):
    id: str
    name: str
    metadata: BlockMetadata | None

    def __new__(cls, *args, **kwargs):  # noqa
        instance = super().__new__(cls)
        instance.id = str(uuid.uuid4())
        return instance


class Clonable(ABC):
    @abstractmethod
    def clone(self, clone_children: Callable) -> Self:
        raise NotImplementedError


@dataclass
class BlockMetadata:
    project_name: str
    project_hyperparameters: dict | None
    fold_index: int
    target: str
    inference: bool
    preprocessing_max_memory_size: int | None


class Composite(Block, Clonable, ABC):
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
        artifacts_length_should_match: bool = (
            True  # returned Artifacts should be the same length as the input
        )

    properties: Properties

    @abstractmethod
    def get_children_primary(self, only_traversal: bool) -> Pipelines:
        raise NotImplementedError

    def get_children_secondary(
        self,
    ) -> Pipelines | None:
        return None

    @abstractmethod
    def postprocess_result_primary(
        self,
        results: list[pd.DataFrame],
        y: pd.Series | None,
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def postprocess_result_secondary(
        self,
        primary_results: pd.DataFrame,
        secondary_results: list[pd.DataFrame],
        y: pd.Series | None,
        in_sample: bool,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def before_fit(self, X: pd.DataFrame) -> None:
        pass

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> tuple[pd.DataFrame, T, Artifact]:
        return X, y, artifact

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        artifact: Artifact,
        results_primary: pd.DataFrame,
        index: int,
        fit: bool,
    ) -> tuple[pd.DataFrame, T, Artifact]:
        return X, y, artifact

    def postprocess_artifacts_primary(
        self,
        primary_artifacts: list[Artifact],
        results: list[pd.DataFrame],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return concat_on_columns_with_duplicates(
            primary_artifacts, strategy=ResolutionStrategy.last
        )

    def postprocess_artifacts_secondary(
        self,
        primary_artifacts: pd.DataFrame,
        secondary_artifacts: list[Artifact],
        original_artifact: Artifact,
    ) -> pd.DataFrame:
        return concat_on_columns_with_duplicates(
            [primary_artifacts, *secondary_artifacts], strategy=ResolutionStrategy.last
        )


class Sampler(Block, Clonable, ABC):
    @abstractmethod
    def get_children_primary(self, only_traversal: bool) -> Pipelines:
        raise NotImplementedError

    def before_fit(self, X: pd.DataFrame) -> None:
        pass

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> tuple[pd.DataFrame, T, Artifact]:
        return X, y, artifact


class Optimizer(Block, Clonable, ABC):
    splitter: Splitter
    metadata: Composite.Metadata | None
    backend: Backend | None

    @abstractmethod
    def get_candidates(self, only_traversal: bool) -> list[Pipeline]:
        """
        Called iteratively, until an array with a length of zero is returned.
        Then the loop finishes the candidate evaluation process.
        """
        raise NotImplementedError

    @abstractmethod
    def get_optimized_pipeline(self) -> Pipeline | None:
        raise NotImplementedError

    @abstractmethod
    def process_candidate_results(
        self,
        results: list[pd.DataFrame],
        y: pd.Series,
        artifacts: list[pd.DataFrame],
    ) -> Artifact | None:
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

        requires_X: bool
        memory_size: int | float | None = None  # During the in_sample period, memory will contain all data. During inference, if not `None`, it will inject past window with size of `memory` to update() & transformation().
        disable_memory: bool = True  # If `True`, memory will not be used at all, even if `memory_size` is not `None`.
        model_type: ModelType | None = None
        _internal_supports_minibatch_backtesting: bool = False  # internal, during backtesting, calls predict_in_sample() instead of predict()

    @dataclass
    class State:
        memory_X: pd.DataFrame
        memory_y: pd.Series
        memory_sample_weights: pd.Series | None

    properties: Properties
    name: str
    _state: State | None

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        raw_y: pd.Series | None = None,
    ) -> Artifact | None:
        """
        Called once, with on initial training window.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        raw_y: pd.Series | None = None,
    ) -> Artifact | None:
        """
        Subsequent calls to update the model.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        raise NotImplementedError


class InvertibleTransformation(Transformation, ABC):
    @abstractmethod
    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        raise NotImplementedError


class Tunable(ABC):
    params_to_try: dict | None

    def get_params(self) -> dict:
        """
        The default implementation assumes that:
        1. All init parameters are stored on the object as property (with the same name/key).
        2. There are no modifications/conversions of the init parameters that'd prevent them from being used again (reconstructing the object from them).
        """
        return get_initialization_parameters(self)

    def get_params_to_try(self) -> dict | None:
        return self.params_to_try

    def clone_with_params(
        self, parameters: dict, clone_children: Callable | None = None
    ) -> Tunable:
        """
        The default implementation only works for Transformations, when parameters and the init parameters match 100%.
        """
        return self.__class__(**parameters)


class FeatureSelector(Transformation):
    selected_features: list[str]


Pipeline = Block | Sequence[Block]
"""A list of `fold` objects that are executed sequentially. Or a single object."""
Pipelines = Sequence[Pipeline]
"""Multiple, independent `Pipeline`s."""

TransformationPipeline = (
    Transformation | Composite | Sequence[Transformation | Composite]
)


TrainedPipeline = Pipeline
TrainedPipelines = list[pd.Series]
"""A list of trained `Pipeline`s, to be used for backtesting."""
OutOfSamplePredictions = pd.DataFrame
"""The backtest's resulting out-of-sample output."""
InSamplePredictions = pd.DataFrame
"""The backtest's resulting in-sample output."""


@dataclass
class PipelineCard:
    preprocessing: TransformationPipeline | None
    pipeline: Pipeline
    event_labeler: Labeler | None = None
    event_filter: EventFilter | None = None
    project_name: str | None = None
    project_hyperparameters: dict | None = None
    trim_initial_period_after_preprocessing: bool = False  # If `True`, the initial period (determined by the memory_size of transformations in the preprocessing pipeline) will be trimmed after preprocessing, but before the pipeline is executed.


@dataclass
class TrainedPipelineCard:
    project_name: str
    preprocessing: TrainedPipelines
    pipeline: TrainedPipelines
    event_labeler: Labeler | None
    event_filter: EventFilter | None
    project_hyperparameters: dict | None
    trim_initial_period_after_preprocessing: bool


def fit_noop(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: pd.Series | None = None,
    raw_y: pd.Series | None = None,
) -> None:
    pass


class SingleFunctionTransformation(Transformation):
    properties = Transformation.Properties(requires_X=True)

    def get_function(self) -> Callable:
        raise NotImplementedError

    fit = fit_noop
    update = fit_noop

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return concat_on_columns([X, self.get_function()(X)])


EventDataFrame = pd.DataFrame


class Artifact(pd.DataFrame):
    @staticmethod
    def empty(index: pd.Index) -> Artifact:
        return pd.DataFrame(index=index)  # type: ignore

    @staticmethod
    def dummy_events(index: pd.Index) -> Artifact:
        return pd.DataFrame(
            index=index,
            data={
                "event_start": index,
                "event_end": index,
                "event_label": [1.0] * len(index),
                "event_raw": [9.0] * len(index),
                "event_sample_weights": [1.0] * len(index),
                "event_test_sample_weights": [1.0] * len(index),
            },
        )  # type: ignore

    @staticmethod
    def get_sample_weights(artifact: Artifact) -> pd.Series | None:
        if artifact is None:
            return None
        if "event_sample_weights" not in artifact.columns:
            return None
        return artifact["event_sample_weights"]

    @staticmethod
    def get_test_sample_weights(artifact: Artifact) -> pd.Series | None:
        if "event_test_sample_weights" not in artifact.columns:
            if "sample_weights" in artifact.columns:
                return artifact["sample_weights"]
            return None
        return artifact["event_test_sample_weights"]

    @staticmethod
    def get_raw_y(artifact: Artifact) -> pd.Series | None:
        if artifact is None:
            return None
        if "event_raw" not in artifact.columns:
            return None
        return artifact["event_raw"]

    @staticmethod
    def get_event_label(artifact: Artifact) -> pd.Series | None:
        if "event_label" not in artifact.columns:
            return None
        return artifact["event_label"]

    @staticmethod
    def get_events(artifact: Artifact) -> EventDataFrame:
        if "event_start" not in artifact.columns:
            return None  # type: ignore
        columns = [
            "event_start",
            "event_end",
            "event_label",
            "event_raw",
            "event_sample_weights",
            "event_test_sample_weights",
        ]
        if "event_strategy_label" in artifact.columns:
            columns.append("event_strategy_label")
        return artifact[columns]

    @staticmethod
    def from_events(
        index: pd.Index,
        events: EventDataFrame | None,
    ) -> Artifact:
        if events is not None and not events.columns[0].startswith("event_"):
            events = events.add_prefix("event_")
        if events is None or events.empty:
            return Artifact.empty(index)  # type: ignore
        return events.reindex(index)  # type: ignore

    @staticmethod
    def events_from_data(
        start: pd.DatetimeIndex,
        end: pd.DatetimeIndex,
        label: pd.Series,
        raw: pd.Series,
        strategy_label: pd.Series | None = None,
        sample_weights: pd.Series | None = None,
        test_sample_weights: pd.Series | None = None,
    ) -> EventDataFrame:
        return EventDataFrame(
            data={
                "event_start": start,
                "event_end": end,
                "event_label": label,
                "event_raw": raw,
                "event_sample_weights": pd.Series(1.0, index=start)
                if sample_weights is None
                else sample_weights,
                "event_test_sample_weights": pd.Series(1.0, index=start)
                if test_sample_weights is None
                else test_sample_weights,
            }
            | (
                {"event_strategy_label": strategy_label}
                if strategy_label is not None
                else {}
            ),
        )


class PredefinedFunction(ParsableEnum):
    mean = "mean"
    sum = "sum"
    median = "median"
    std = "std"
    var = "var"
    kurt = "kurt"
    min = "min"
    max = "max"
    corr = "corr"
    cov = "cov"
    skew = "skew"
    sem = "sem"


class EventFilter(ABC):
    @abstractmethod
    def get_event_start_times(self, y: pd.Series) -> pd.DatetimeIndex:
        raise NotImplementedError


class Labeler(ABC):
    @abstractmethod
    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_all_possible_labels(self) -> list[int]:
        raise NotImplementedError


class LabelingStrategy(ABC):
    @abstractmethod
    def label(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def get_all_labels(self) -> list[int]:
        raise NotImplementedError


class WeightingStrategy(ABC):
    @abstractmethod
    def calculate(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError


@dataclass
class Backend:
    name: str
    process_child_transformations: Callable
    train_pipeline: Callable
    backtest_pipeline: Callable

    def __init__(self):  # to be able to initalize it with the default parameters
        pass
