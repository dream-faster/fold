from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class EventFilter(ABC):
    @abstractmethod
    def get_event_start_times(self, y: pd.Series) -> pd.DatetimeIndex:
        raise NotImplementedError

    # TODO: this is the online, out-of-sample equivalent of get_event_start_times, need a final name
    def update(self, y: pd.Series) -> bool:
        raise NotImplementedError


# class EventSchema(pa.SchemaModel):
#     start: Series[pd.Timestamp]
#     end: Series[pd.Timestamp]
#     label: Series[int]
#     returns: Series[float]


class EventLabeler(ABC):
    @abstractmethod
    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> pd.DataFrame:
        raise NotImplementedError

    def get_all_possible_labels(self) -> List[int]:
        raise NotImplementedError
