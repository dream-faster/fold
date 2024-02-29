# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.
from __future__ import annotations

import pandas as pd

from ..base import EventDataFrame, EventFilter, Labeler
from .labeling import *  # noqa: F403
from .weights import *  # noqa: F403


def _create_events(
    y: pd.Series,
    event_filter: EventFilter | None,
    labeler: Labeler | None,
) -> EventDataFrame | None:
    if labeler is None:
        return None
    start_times = (
        event_filter.get_event_start_times(y) if event_filter is not None else y.index
    )
    print(f"Filtered out {(1 - (len(start_times) / len(y))) * 100}% timestamps.")

    return labeler.label_events(start_times, y).reindex(y.index)  # type: ignore
