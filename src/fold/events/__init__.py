# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.
from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from fold.base.classes import TrainedPipelineCard

from ..base import (
    Artifact,
    Composite,
    EventDataFrame,
    EventFilter,
    Labeler,
    LabelingStrategy,
    Pipeline,
    PipelineCard,
    Pipelines,
    T,
    Transformation,
    WeightingStrategy,
    fit_noop,
)
from ..utils.dataframe import (
    ResolutionStrategy,
    concat_on_columns,
    concat_on_columns_with_duplicates,
)
from ..utils.list import wrap_in_double_list_if_needed, wrap_in_list
from .filters import EveryNth, FilterZero, NoFilter
from .labeling import *  # noqa
from .weights import *  # noqa


class UsePredefinedEvents(Composite):
    name = "UsePredefinedEvents"

    def __init__(
        self,
        wrapped_pipeline: Pipeline,
    ) -> None:
        self.wrapped_pipeline = wrap_in_double_list_if_needed(wrapped_pipeline)
        self.properties = Composite.Properties(
            primary_only_single_pipeline=True, artifacts_length_should_match=False
        )
        self.metadata = None

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
            events.event_label,
            events,
        )

    def clone(self, clone_children: Callable) -> UsePredefinedEvents:
        clone = UsePredefinedEvents(
            clone_children(self.wrapped_pipeline),
        )
        clone.properties = self.properties
        clone.name = self.name
        clone.metadata = self.metadata
        return clone


def _create_events(
    y: pd.Series, pipeline_card: Union[PipelineCard, TrainedPipelineCard]
) -> Optional[EventDataFrame]:
    if pipeline_card.event_labeler is None:
        return None
    start_times = (
        pipeline_card.event_filter.get_event_start_times(y)
        if pipeline_card.event_filter is not None
        else y.index
    )
    return pipeline_card.event_labeler.label_events(start_times, y).reindex(y.index)
