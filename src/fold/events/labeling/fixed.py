from __future__ import annotations

from logging import getLogger
from typing import Callable, List, Optional, Union

import pandas as pd

from ...base import (
    EventDataFrame,
    Labeler,
    LabelingStrategy,
    PredefinedFunction,
    WeightingStrategy,
)
from ...utils.forward import create_forward_rolling
from ..weights import NoWeighting, WeightBySumWithLookahead

logger = getLogger("fold:labeling")


class FixedForwardHorizon(Labeler):
    time_horizon: int

    def __init__(
        self,
        time_horizon: int,
        labeling_strategy: LabelingStrategy,
        weighting_strategy: Optional[WeightingStrategy],
        weighting_strategy_test: WeightingStrategy = WeightBySumWithLookahead(),
        transformation_function: Optional[Callable] = None,
        aggregate_function: Union[
            str, PredefinedFunction, Callable
        ] = PredefinedFunction.sum,
        flip_sign: bool = False,
        shift_by: Optional[int] = None,
    ):
        self.time_horizon = time_horizon
        self.labeling_strategy = labeling_strategy
        self.weighting_strategy = (
            weighting_strategy if weighting_strategy else NoWeighting()
        )
        self.weighting_strategy_test = weighting_strategy_test
        self.transformation_function = transformation_function
        self.aggregate_function = (
            aggregate_function
            if isinstance(aggregate_function, Callable)
            else getattr(
                pd.core.window.rolling.Rolling,
                PredefinedFunction.from_str(aggregate_function).value,
            )
        )
        self.flip_sign = flip_sign
        self.shift_by = shift_by

    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        forward_rolling_aggregated = create_forward_rolling(
            transformation_func=self.transformation_function,
            agg_func=self.aggregate_function,
            series=y,
            period=self.time_horizon,
            shift_by=self.shift_by,
        )
        cutoff_point = y.index[-self.time_horizon]
        event_start_times = event_start_times[event_start_times < cutoff_point]
        forward_rolling_aggregated = forward_rolling_aggregated[
            event_start_times
        ]  # filter based on events

        labels = self.labeling_strategy.label(forward_rolling_aggregated)
        sample_weights = self.weighting_strategy.calculate(forward_rolling_aggregated)
        test_sample_weights = self.weighting_strategy_test.calculate(
            forward_rolling_aggregated
        )

        if self.flip_sign:
            if len(labels.dropna().unique()) == 3:
                labels = labels * -1
            elif len(labels.dropna().unique()) == 2:
                labels = 1 - labels
            else:
                logger.warn(
                    f"{len(labels.dropna().unique())} classes detected, can't flip sign"
                )

        offset = pd.Timedelta(value=self.time_horizon, unit=y.index.freqstr)
        return EventDataFrame.from_data(
            start=event_start_times,
            end=(event_start_times + offset).astype("datetime64[s]"),
            label=labels,
            raw=forward_rolling_aggregated,
            sample_weights=sample_weights,
            test_sample_weights=test_sample_weights,
        )

    def get_labels(self) -> List[int]:
        return self.labeling_strategy.get_all_labels()
