import pandas as pd

from ...base import EventFilter


class EveryNth(EventFilter):
    def __init__(self, n: int):
        self.n = n

    def get_event_start_times(self, y: pd.Series) -> pd.DatetimeIndex:
        return y.index[:: self.n]
