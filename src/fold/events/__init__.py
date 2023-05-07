# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..base import Artifact, Composite, Pipeline, Pipelines, T
from ..utils.list import wrap_in_list
from .base import EventFilter, Labeler
from .filters import EveryNth, NoFilter
from .labeling import BinarizeFixedForwardHorizon


class CreateEvents(Composite):
    properties = Composite.Properties(primary_only_single_pipeline=True)
    name = "CreateEvents"

    def __init__(
        self,
        wrapped_pipeline: Pipeline,
        labeler: Labeler,
        event_filter: EventFilter,
    ) -> None:
        self.wrapped_pipeline = wrap_in_list(wrapped_pipeline)
        self.labeler = labeler
        self.filter = event_filter

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        original_start_times = self.filter.get_event_start_times(y)
        events = self.labeler.label_events(original_start_times, y)
        self.last_events = events
        return X.loc[events.index], events["label"]

    def get_child_transformations_primary(self) -> Pipelines:
        return self.wrapped_pipeline

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return results[0]

    def postprocess_artifacts_primary(self, artifacts: List[Artifact]) -> pd.DataFrame:
        return self.last_events

    def clone(self, clone_child_transformations: Callable) -> CreateEvents:
        clone = CreateEvents(
            clone_child_transformations(self.wrapped_pipeline),
            self.labeler,
            self.filter,
        )
        clone.properties = self.properties
        return clone
