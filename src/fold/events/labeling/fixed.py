from __future__ import annotations

from typing import Callable, List, Optional, Union

import pandas as pd

from ...base import PredefinedFunction
from ...utils.forward import create_forward_rolling
from ..base import EventDataFrame, Labeler, LabelingStrategy, WeighingStrategy
from ..weights import NoWeighing, WeightBySumWithLookahead


class FixedForwardHorizon(Labeler):
    time_horizon: int

    def __init__(
        self,
        time_horizon: int,
        labeling_strategy: LabelingStrategy,
        weighing_strategy: Optional[WeighingStrategy],
        weighing_strategy_test: WeighingStrategy = WeightBySumWithLookahead(),
        aggregate_function: Union[
            PredefinedFunction, Callable
        ] = PredefinedFunction.sum,
    ):
        self.time_horizon = time_horizon
        self.labeling_strategy = labeling_strategy
        self.weighing_strategy = (
            weighing_strategy if weighing_strategy else NoWeighing()
        )
        self.weighing_strategy_test = weighing_strategy_test
        self.aggregate_function = (
            aggregate_function
            if isinstance(aggregate_function, Callable)
            else getattr(
                pd.core.window.rolling.Rolling,
                PredefinedFunction.from_str(aggregate_function).value,
            )
        )

    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        forward_rolling_aggregated = create_forward_rolling(
            self.aggregate_function, y, self.time_horizon
        )
        cutoff_point = y.index[-self.time_horizon]
        event_start_times = event_start_times[event_start_times < cutoff_point]
        event_candidates = forward_rolling_aggregated[event_start_times]

        labels = self.labeling_strategy.label(event_candidates)
        raw_returns = forward_rolling_aggregated[event_start_times]
        sample_weights = self.weighing_strategy.calculate(raw_returns)
        test_sample_weights = self.weighing_strategy_test.calculate(raw_returns)

        offset = pd.Timedelta(value=self.time_horizon, unit=y.index.freqstr)
        return EventDataFrame.from_data(
            start=event_start_times,
            end=event_start_times + offset,
            label=labels,
            raw=raw_returns,
            sample_weights=sample_weights,
            test_sample_weights=test_sample_weights,
        )

    def get_labels(self) -> List[int]:
        return self.labeling_strategy.get_all_labels()
