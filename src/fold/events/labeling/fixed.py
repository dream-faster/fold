from typing import List

import pandas as pd

from ...utils.forward import create_forward_rolling_sum
from ..base import EventDataFrame, Labeler


class BinarizeFixedForwardHorizon(Labeler):
    time_horizon: int

    def __init__(self, time_horizon: int):
        self.time_horizon = time_horizon

    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        forward_rolling_sum = create_forward_rolling_sum(y, self.time_horizon)
        cutoff_point = y.index[-self.time_horizon]
        event_start_times = event_start_times[event_start_times < cutoff_point]
        event_candidates = forward_rolling_sum[event_start_times]

        def map_to_binary(series: pd.Series) -> pd.Series:
            series.loc[series >= 0.0] = 1
            series.loc[series < 0.0] = -1
            return series

        labels = map_to_binary(event_candidates)

        offset = pd.Timedelta(value=self.time_horizon, unit=y.index.freqstr)
        events = EventDataFrame(
            start=event_start_times,
            end=event_start_times + offset,
            label=labels,
            raw=forward_rolling_sum[event_start_times],
        )
        return events

    def get_labels(self) -> List[int]:
        return [-1, 1]
