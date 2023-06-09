from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ...utils.forward import create_forward_rolling_sum
from ..base import EventDataFrame, Labeler, LabelingStrategy, WeighingStrategy
from ..weights import NoWeighing, WeightByMaxWithLookahead


class FixedForwardHorizon(Labeler):
    time_horizon: int

    def __init__(
        self,
        time_horizon: int,
        labeling_strategy: LabelingStrategy,
        weighing_strategy: Optional[WeighingStrategy],
        weighing_strategy_test: WeighingStrategy = WeightByMaxWithLookahead(),
    ):
        self.time_horizon = time_horizon
        self.labeling_strategy = labeling_strategy
        self.weighing_strategy = (
            weighing_strategy if weighing_strategy else NoWeighing()
        )
        self.weighing_strategy_test = weighing_strategy_test

    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        forward_rolling_sum = create_forward_rolling_sum(y, self.time_horizon)
        cutoff_point = y.index[-self.time_horizon]
        event_start_times = event_start_times[event_start_times < cutoff_point]
        event_candidates = forward_rolling_sum[event_start_times]

        labels = self.labeling_strategy.label(event_candidates)
        raw_returns = forward_rolling_sum[event_start_times]
        sample_weights = self.weighing_strategy.calculate(raw_returns)
        test_sample_weights = self.weighing_strategy_test.calculate(raw_returns)

        offset = pd.Timedelta(value=self.time_horizon, unit=y.index.freqstr)
        events = EventDataFrame(
            start=event_start_times,
            end=event_start_times + offset,
            label=labels,
            raw=raw_returns,
            sample_weights=sample_weights,
            test_sample_weights=test_sample_weights,
        )
        return events

    def get_labels(self) -> List[int]:
        return self.labeling_strategy.get_all_labels()
