# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..base import Artifact, Composite, Pipeline, Pipelines, T, Transformation, fit_noop
from ..utils.dataframe import (
    ResolutionStrategy,
    concat_on_columns,
    concat_on_columns_with_duplicates,
)
from ..utils.list import wrap_in_double_list_if_needed, wrap_in_list
from .base import EventFilter, Labeler, LabelingStrategy, WeighingStrategy
from .filters import EveryNth, FilterZero, NoFilter
from .labeling import *  # noqa
from .weights import *  # noqa


def CreateEvents(
    wrapped_pipeline: Pipeline,
    labeler: Labeler,
    filter: EventFilter = NoFilter(),
) -> _CreateEvents:
    return _CreateEvents(wrapped_pipeline, _EventLabelWrapper(filter, labeler))


class _CreateEvents(Composite):
    properties = Composite.Properties(primary_only_single_pipeline=True)
    name = "CreateEvents"
    transformation: List[_EventLabelWrapper]

    def __init__(
        self,
        wrapped_pipeline: Pipeline,
        event_label_wrapper: _EventLabelWrapper,
    ) -> None:
        self.wrapped_pipeline = wrap_in_double_list_if_needed(wrapped_pipeline)
        self.transformation = wrap_in_list(event_label_wrapper)

    def get_children_primary(self) -> Pipelines:
        return self.transformation

    def get_children_secondary(self) -> Optional[Pipelines]:
        return self.wrapped_pipeline

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        artifact: Artifact,
        results_primary: List[pd.DataFrame],
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        events = results_primary[0].dropna()
        return (X.loc[events.index], events["label"], events)

    def postprocess_result_secondary(
        self,
        primary_results: List[pd.DataFrame],
        secondary_results: List[pd.DataFrame],
        y: Optional[pd.Series],
        in_sample: bool,
    ) -> pd.DataFrame:
        return secondary_results[0].reindex(y.index)

    def postprocess_artifacts_primary(
        self,
        primary_artifacts: List[Artifact],
        results: List[pd.DataFrame],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return results[0]

    def postprocess_artifacts_secondary(
        self,
        primary_artifacts: pd.DataFrame,
        secondary_artifacts: List[Artifact],
        original_artifact: Artifact,
    ) -> pd.DataFrame:
        return concat_on_columns_with_duplicates(
            [primary_artifacts, concat_on_columns(secondary_artifacts)],
            strategy=ResolutionStrategy.last,
        )

    def clone(self, clone_children: Callable) -> _CreateEvents:
        clone = _CreateEvents(
            clone_children(self.wrapped_pipeline),
            clone_children(self.transformation),
        )
        clone.properties = self.properties
        clone.name = self.name
        return clone


class UsePredefinedEvents(Composite):
    properties = Composite.Properties(
        primary_only_single_pipeline=True, artifacts_length_should_match=False
    )
    name = "UsePredefinedEvents"

    def __init__(
        self,
        wrapped_pipeline: Pipeline,
    ) -> None:
        self.wrapped_pipeline = wrap_in_double_list_if_needed(wrapped_pipeline)

    def get_children_primary(self) -> Pipelines:
        return self.wrapped_pipeline

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        events = Artifact.get_events(artifact)
        if events is None:
            raise ValueError(
                "You need to pass in `events` for `UsePredefinedEvents` to use when calling train() / backtest()."
            )
        events = events.dropna()
        return (
            X.loc[events.index],
            events.label,
            events,
        )

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return results[0].reindex(y.index)

    def postprocess_artifacts_primary(
        self,
        primary_artifacts: List[Artifact],
        results: List[pd.DataFrame],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return concat_on_columns_with_duplicates(
            primary_artifacts,
            strategy=ResolutionStrategy.last,
        )

    def clone(self, clone_children: Callable) -> UsePredefinedEvents:
        clone = UsePredefinedEvents(
            clone_children(self.wrapped_pipeline),
        )
        clone.properties = self.properties
        clone.name = self.name
        return clone


class _EventLabelWrapper(Transformation):
    name = "EventLabelWrapper"
    properties = Transformation.Properties(
        mode=Transformation.Properties.Mode.online,
        requires_X=False,
        _internal_supports_minibatch_backtesting=True,
        memory_size=1,
    )
    last_events: Optional[pd.DataFrame] = None

    def __init__(self, event_filter: EventFilter, labeler: Labeler) -> None:
        self.filter = event_filter
        self.labeler = labeler
        self.name = (
            f"{self.filter.__class__.__name__}-{self.labeler.__class__.__name__}"
        )

    fit = fit_noop
    update = fit

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        past_y = self._state.memory_y
        original_start_times = self.filter.get_event_start_times(past_y)
        events = self.labeler.label_events(original_start_times, past_y)
        if in_sample is False and events.dropna().empty:
            self.properties.memory_size += 1
        elif in_sample is False and not events.dropna().empty:
            self.properties.memory_size = 1
        return events.reindex(X.index), None
