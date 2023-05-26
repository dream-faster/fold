from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TypeVar, Union

import pandas as pd

from ..splitters import SingleWindowSplitter
from ..utils.introspection import get_initialization_parameters

T = TypeVar("T", Optional[pd.Series], pd.Series)
X = pd.DataFrame
Artifact = pd.DataFrame


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
        self, X: pd.DataFrame, index: int, y: T, extras: Extras, fit: bool
    ) -> Tuple[pd.DataFrame, T, Extras]:
        return X, y, extras

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        results_primary: List[pd.DataFrame],
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T]:
        return X, y

    def postprocess_artifacts_primary(
        self, artifacts: List[Artifact], extras: Extras
    ) -> pd.DataFrame:
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
        extras: Extras,
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

    def __init__(
        self,
        start: pd.DatetimeIndex,
        end: pd.DatetimeIndex,
        label: pd.Series,
        raw: pd.Series,
    ):
        super().__init__(data={"start": start, "end": end, "label": label, "raw": raw})


@dataclass
class Extras:
    events: Optional[EventDataFrame]
    sample_weights: Optional[pd.Series]

    def __init__(
        self,
        events: Optional[EventDataFrame] = None,
        sample_weights: Optional[pd.Series] = None,
    ):
        self.events = events
        self.sample_weights = sample_weights

    def loc(self, s) -> Extras:
        return Extras(
            events=self.events.loc[s] if self.events is not None else None,
            sample_weights=self.sample_weights.loc[s]
            if self.sample_weights is not None
            else None,
        )

    def iloc(self, s) -> Extras:
        return Extras(
            events=self.events.iloc[s] if self.events is not None else None,
            sample_weights=self.sample_weights.iloc[s]
            if self.sample_weights is not None
            else None,
        )

    def __len__(self) -> int:
        return len(self.events) if self.events is not None else 0
