from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class EventDataFrame(pd.DataFrame):
    start: pd.Series
    end: pd.Series
    label: pd.Series
    raw: pd.Series

    def __init__(
        self,
        start: pd.DatetimeIndex,
        end: pd.DatetimeIndex,
        label: pd.Series,
        raw: pd.Series,
    ):
        super().__init__({"start": start, "end": end, "label": label, "raw": raw})


class EventFilter(ABC):
    @abstractmethod
    def get_event_start_times(self, y: pd.Series) -> pd.DatetimeIndex:
        raise NotImplementedError

    # TODO: this is the online, out-of-sample equivalent of get_event_start_times, need a final name
    def update(self, y: pd.Series) -> bool:
        raise NotImplementedError


class Labeler(ABC):
    @abstractmethod
    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        raise NotImplementedError

    def get_all_possible_labels(self) -> List[int]:
        raise NotImplementedError
