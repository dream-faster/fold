from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import pandas as pd
from fold_extensions.labeling.sample_weight import calculate_sample_weights

from fold.events.base import EventDataFrame, Labeler
from fold.utils.forward import create_forward_rolling_sum


class LabelingStrategy(ABC):
    @abstractmethod
    def label(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def get_all_labels(self) -> List[int]:
        raise NotImplementedError


class BinarizeSign(LabelingStrategy):
    def label(self, series: pd.Series) -> pd.Series:
        labels = series.copy()
        labels.loc[series >= 0.0] = 1
        labels.loc[series < 0.0] = 0
        return labels

    def get_all_labels(self) -> List[int]:
        return [0, 1]


class Noop(LabelingStrategy):
    def label(self, series: pd.Series) -> pd.Series:
        return series

    def get_all_labels(self) -> List[int]:
        return [0, 1]


class FixedForwardHorizon(Labeler):
    time_horizon: int

    def __init__(
        self,
        time_horizon: int,
        strategy: LabelingStrategy,
        sample_weights_window: Optional[Union[float, int]] = 0.2,
    ):
        self.time_horizon = time_horizon
        self.strategy = strategy
        self.sample_weights_window = sample_weights_window

    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        forward_rolling_sum = create_forward_rolling_sum(y, self.time_horizon)
        cutoff_point = y.index[-self.time_horizon]
        event_start_times = event_start_times[event_start_times < cutoff_point]
        event_candidates = forward_rolling_sum[event_start_times]

        labels = self.strategy.label(event_candidates)
        raw_returns = forward_rolling_sum[event_start_times]
        sample_weights = calculate_sample_weights(
            raw_returns, self.sample_weights_window
        )

        offset = pd.Timedelta(value=self.time_horizon, unit=y.index.freqstr)
        events = EventDataFrame(
            start=event_start_times,
            end=event_start_times + offset,
            label=labels,
            raw=raw_returns,
            sample_weights=sample_weights,
        )
        return events

    def get_labels(self) -> List[int]:
        return self.strategy.get_all_labels()
