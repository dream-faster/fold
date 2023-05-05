import pandas as pd

from ...utils.forward import create_forward_rolling_sum
from ..base import EventLabeler


class BinarizeFixedForwardHorizon(EventLabeler):
    time_horizon: int

    def __init__(self, time_horizon: int):
        self.time_horizon = time_horizon

    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> pd.DataFrame:
        forward_rolling_sum = create_forward_rolling_sum(y, self.time_horizon)
        cutoff_point = y.index[-self.time_horizon]
        event_start_times = event_start_times[event_start_times < cutoff_point]
        event_candidates = forward_rolling_sum[event_start_times]

        def get_class_binary(x: float) -> int:
            return -1 if x <= 0.0 else 1

        # TODO: this can be done much more efficiently
        labels = event_candidates.map(get_class_binary)
        offset = pd.Timedelta(value=self.time_horizon, unit=y.index.freqstr)
        events = pd.DataFrame(
            {
                "start": event_start_times,
                "end": event_start_times + offset,
                "label": labels,
                "raw": forward_rolling_sum[event_start_times],
            }
        )
        return events

    def get_labels(self) -> list[int]:
        return [-1, 1]
