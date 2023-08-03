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
    WeightingStrategy,
)
from ..utils.list import wrap_in_double_list_if_needed
from .filters import EveryNth, FilterZero, NoFilter
from .labeling import *  # noqa
from .weights import *  # noqa


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
