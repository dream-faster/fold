import pandas as pd

from ...base import EventFilter


class NoFilter(EventFilter):
    def get_event_start_times(self, y: pd.Series) -> pd.DatetimeIndex:
        return y.index


class FilterZero(EventFilter):
    def get_event_start_times(self, y: pd.Series) -> pd.DatetimeIndex:
        return y[y != 0.0].index
