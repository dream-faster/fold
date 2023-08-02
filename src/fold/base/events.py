from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from .classes import EventDataFrame


class EventFilter(ABC):
    @abstractmethod
    def get_event_start_times(self, y: pd.Series) -> pd.DatetimeIndex:
        raise NotImplementedError


class Labeler(ABC):
    @abstractmethod
    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        raise NotImplementedError

    def get_all_possible_labels(self) -> List[int]:
        raise NotImplementedError


class LabelingStrategy(ABC):
    @abstractmethod
    def label(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def get_all_labels(self) -> List[int]:
        raise NotImplementedError


class WeightingStrategy(ABC):
    @abstractmethod
    def calculate(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError
