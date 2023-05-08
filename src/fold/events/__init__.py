# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..base import Artifact, Composite, Pipeline, Pipelines, T, Transformation, fit_noop
from ..utils.list import wrap_in_list
from .base import EventFilter, Labeler
from .filters import EveryNth, NoFilter
from .labeling import BinarizeFixedForwardHorizon


def CreateEvents(
    wrapped_pipeline: Pipeline,
    labeler: Labeler,
    filter: EventFilter = NoFilter(),
) -> _CreateEvents:
    return _CreateEvents(wrapped_pipeline, _EventLabelWrapper(filter, labeler))


class _CreateEvents(Composite):
    properties = Composite.Properties(primary_only_single_pipeline=True)
    name = "_CreateEvents"
    transformation: List[_EventLabelWrapper]

    def __init__(
        self,
        wrapped_pipeline: Pipeline,
        event_label_wrapper: _EventLabelWrapper,
    ) -> None:
        self.wrapped_pipeline = wrap_in_list(wrapped_pipeline)
        self.transformation = wrap_in_list(event_label_wrapper)

    def get_child_transformations_primary(self) -> Pipelines:
        return self.transformation

    def get_child_transformations_secondary(self) -> Optional[Pipelines]:
        return self.wrapped_pipeline

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        results_primary: List[pd.DataFrame],
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T]:
        events = results_primary[0].dropna()
        return X.loc[events.index], events["label"]

    def postprocess_result_secondary(
        self,
        primary_results: List[pd.DataFrame],
        secondary_results: List[pd.DataFrame],
        y: Optional[pd.Series],
        in_sample: bool,
    ) -> pd.DataFrame:
        return secondary_results[0].reindex(y.index)

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

    def clone(self, clone_child_transformations: Callable) -> _CreateEvents:
        clone = _CreateEvents(
            clone_child_transformations(self.wrapped_pipeline),
            clone_child_transformations(self.transformation),
        )
        clone.properties = self.properties
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
        return events["label"].to_frame().reindex(X.index), events
