from __future__ import annotations

from typing import Callable, List, Optional, Union

import pandas as pd

from ...base import PredefinedFunction
from ...utils.forward import create_forward_rolling
from ..base import EventDataFrame, Labeler, LabelingStrategy, WeightingStrategy
from ..weights import NoWeighting, WeightBySumWithLookahead


class FixedForwardHorizon(Labeler):
    time_horizon: int

    def __init__(
        self,
        time_horizon: int,
        labeling_strategy: LabelingStrategy,
        weighting_strategy: Optional[WeightingStrategy],
        weighting_strategy_test: WeightingStrategy = WeightBySumWithLookahead(),
        aggregate_function: Union[
            str, PredefinedFunction, Callable
        ] = PredefinedFunction.sum,
    ):
        self.time_horizon = time_horizon
        self.labeling_strategy = labeling_strategy
        self.weighting_strategy = (
            weighting_strategy if weighting_strategy else NoWeighting()
        )
        self.weighting_strategy_test = weighting_strategy_test
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
        sample_weights = self.weighting_strategy.calculate(raw_returns)
        test_sample_weights = self.weighting_strategy_test.calculate(raw_returns)

        offset = pd.Timedelta(value=self.time_horizon, unit=y.index.freqstr)
        return EventDataFrame.from_data(
            start=event_start_times,
            end=(event_start_times + offset).astype("datetime64[s]"),
            label=labels,
            raw=raw_returns,
            sample_weights=sample_weights,
            test_sample_weights=test_sample_weights,
        )

    def get_labels(self) -> List[int]:
        return self.labeling_strategy.get_all_labels()
