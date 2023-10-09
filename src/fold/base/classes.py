from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TypeVar, Union

import pandas as pd
from typing_extensions import Self

from ..splitters import SingleWindowSplitter
from ..utils.dataframe import ResolutionStrategy, concat_on_columns_with_duplicates
from ..utils.enums import ParsableEnum
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


class Clonable(ABC):
    @abstractmethod
    def clone(self, clone_children: Callable) -> Self:
        raise NotImplementedError


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

    @dataclass
    class Metadata:
        project_name: str
        fold_index: int
        target: str
        inference: bool

    properties: Properties
    metadata: Optional[Metadata]

    @abstractmethod
    def get_children_primary(self) -> Pipelines:
        raise NotImplementedError

    def get_children_secondary(
        self,
    ) -> Optional[Pipelines]:
        return None

    def postprocess_result_primary(
        self,
        results: List[pd.DataFrame],
        y: Optional[pd.Series],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def postprocess_result_secondary(
        self,
        primary_results: pd.DataFrame,
        secondary_results: List[pd.DataFrame],
        y: Optional[pd.Series],
        in_sample: bool,
    ) -> pd.DataFrame:
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
        results_primary: pd.DataFrame,
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        return X, y, artifact

    def postprocess_artifacts_primary(
        self,
        primary_artifacts: List[Artifact],
        results: List[pd.DataFrame],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return concat_on_columns_with_duplicates(
            primary_artifacts, strategy=ResolutionStrategy.last
        )

    def postprocess_artifacts_secondary(
        self,
        primary_artifacts: pd.DataFrame,
        secondary_artifacts: List[Artifact],
        original_artifact: Artifact,
    ) -> pd.DataFrame:
        return concat_on_columns_with_duplicates(
            [primary_artifacts] + secondary_artifacts, strategy=ResolutionStrategy.last
        )


class Sampler(Block, Clonable, ABC):
    @abstractmethod
    def get_children_primary(self) -> Pipelines:
        raise NotImplementedError

    def before_fit(self, X: pd.DataFrame) -> None:
        pass

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        return X, y, artifact


class Optimizer(Block, Clonable, ABC):
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
        updateable: bool = True
        memory_size: Optional[
            Union[int, float]
        ] = None  # During the in_sample period, memory will contain all data. During inference, if not `None`, it will inject past window with size of `memory` to update() & transformation().
        disable_memory: bool = False  # If `True`, memory will not be used at all, even if `memory_size` is not `None`.
        model_type: Optional[ModelType] = None
        _internal_supports_minibatch_backtesting: bool = False  # internal, during backtesting, calls predict_in_sample() instead of predict()

    @dataclass
    class State:
        memory_X: pd.DataFrame
        memory_y: pd.Series
        memory_sample_weights: Optional[pd.Series]

    properties: Properties
    name: str
    _state: Optional[State]

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
    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        raise NotImplementedError


class InvertibleTransformation(Transformation, ABC):
    @abstractmethod
    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        raise NotImplementedError


class Tunable(ABC):
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


Pipeline = Union[Block, List[Block]]
"""A list of `fold` objects that are executed sequentially. Or a single object."""
Pipelines = List[Pipeline]
"""Multiple, independent `Pipeline`s."""

TransformationPipeline = Union[
    Union[Transformation, Composite], List[Union[Transformation, Composite]]
]


TrainedPipeline = Pipeline
TrainedPipelines = List[pd.Series]
"""A list of trained `Pipeline`s, to be used for backtesting."""
OutOfSamplePredictions = pd.DataFrame
"""The backtest's resulting out-of-sample output."""
InSamplePredictions = pd.DataFrame
"""The backtest's resulting in-sample output."""


@dataclass
class PipelineCard:
    preprocessing: Optional[TransformationPipeline]
    pipeline: Pipeline
    event_labeler: Optional[Labeler] = None
    event_filter: Optional[EventFilter] = None
    name: Optional[str] = None


@dataclass
class TrainedPipelineCard:
    name: str
    preprocessing: TrainedPipelines
    pipeline: TrainedPipelines
    event_labeler: Optional[Labeler]
    event_filter: Optional[EventFilter]


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

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return pd.concat([X, self.get_function()(X)], axis="columns")


class EventDataFrame(pd.DataFrame):
    @property
    def start(self) -> pd.Series:
        return self["event_start"]

    @property
    def end(self) -> pd.Series:
        return self["event_end"]

    @property
    def label(self) -> pd.Series:
        return self["event_label"]

    @property
    def raw(self) -> pd.Series:
        return self["event_raw"]

    @property
    def sample_weights(self) -> pd.Series:
        return self["event_sample_weights"]

    @property
    def test_sample_weights(self) -> pd.Series:
        return self["event_test_sample_weights"]

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
        return cls(
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
        )


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
    def get_sample_weights(artifact: Artifact) -> Optional[pd.Series]:
        if "event_sample_weights" not in artifact.columns:
            return None
        return artifact["event_sample_weights"]

    @staticmethod
    def get_test_sample_weights(artifact: Artifact) -> Optional[pd.Series]:
        if "event_test_sample_weights" not in artifact.columns:
            if "sample_weights" in artifact.columns:
                return artifact["sample_weights"]
            else:
                return None
        return artifact["event_test_sample_weights"]

    @staticmethod
    def get_event_label(artifact: Artifact) -> Optional[pd.Series]:
        if "event_label" not in artifact.columns:
            return None
        return artifact["event_label"]

    @staticmethod
    def get_events(artifact: Artifact) -> Optional[EventDataFrame]:
        if "event_start" not in artifact.columns:
            return None
        return EventDataFrame.from_data(
            start=artifact.event_start,
            end=artifact.event_end,
            label=artifact.event_label,
            raw=artifact.event_raw,
            sample_weights=artifact.event_sample_weights,
            test_sample_weights=artifact.event_test_sample_weights,
        )

    @staticmethod
    def from_events(
        index: pd.Index,
        events: Optional[EventDataFrame],
    ) -> Artifact:
        if events is not None and not events.columns[0].startswith("event_"):
            events = events.add_prefix("event_")
        if events is None or events.empty:
            return Artifact.empty(index)  # type: ignore
        else:
            return events.reindex(index)  # type: ignore


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

    def get_all_possible_labels(self) -> List[int]:
        raise NotImplementedError


class LabelingStrategy(ABC):
    @abstractmethod
    def label(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def get_all_labels(self) -> List[int]:
        raise NotImplementedError


class WeightingStrategy(ABC):
    @abstractmethod
    def calculate(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError
