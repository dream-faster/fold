from __future__ import annotations

from collections.abc import Callable
from logging import getLogger

import pandas as pd

from ...base import (
    Artifact,
    EventDataFrame,
    Labeler,
    LabelingStrategy,
    PredefinedFunction,
    WeightingStrategy,
)
from ...utils.forward import create_forward_rolling
from ..weights import NoWeighting, WeightByMaxWithLookahead

logger = getLogger("fold:labeling")


class FixedForwardHorizon(Labeler):
    def __init__(
        self,
        time_horizon: int,
        labeling_strategy: LabelingStrategy,
        weighting_strategy: WeightingStrategy | None,
        weighting_strategy_test: WeightingStrategy = WeightByMaxWithLookahead(),  # noqa
        transformation_function: Callable | None = None,
        post_aggregation_transformation_function: Callable | None = None,
        aggregate_function: str
        | PredefinedFunction
        | Callable = PredefinedFunction.sum,
        truncate_if_timeframe_is_smaller: bool = True,
        flip_sign: bool = False,
        shift_by: int | None = None,
    ):
        self.time_horizon = time_horizon
        self.labeling_strategy = labeling_strategy
        self.weighting_strategy = (
            weighting_strategy if weighting_strategy else NoWeighting()
        )
        self.weighting_strategy_test = weighting_strategy_test
        self.transformation_function = transformation_function
        self.post_aggregation_transformation_function = (
            post_aggregation_transformation_function
        )
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
        self.truncate_if_timeframe_is_smaller = truncate_if_timeframe_is_smaller

    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        forward_rolling_aggregated = create_forward_rolling(
            transformation_func=self.transformation_function,
            agg_func=self.aggregate_function,
            series=y,
            period=self.time_horizon,
            extra_shift_by=self.shift_by,
        )
        if self.truncate_if_timeframe_is_smaller:
            cutoff_point = y.index[-self.time_horizon]
            event_start_times = event_start_times[event_start_times < cutoff_point]

        forward_rolling_aggregated = forward_rolling_aggregated[
            event_start_times
        ]  # filter based on events
        if self.post_aggregation_transformation_function:
            forward_rolling_aggregated = self.post_aggregation_transformation_function(
                forward_rolling_aggregated
            )

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
                logger.warning(
                    f"{len(labels.dropna().unique())} classes detected, can't flip sign"
                )

        end_times = y.index[
            (y.index.get_indexer(event_start_times) + self.time_horizon).clip(
                max=len(y) - 1
            )
        ]
        return Artifact.events_from_data(
            start=event_start_times,
            end=end_times,
            label=labels,
            raw=y.loc[labels.index],
            sample_weights=sample_weights,
            test_sample_weights=test_sample_weights,
        )

    def get_all_possible_labels(self) -> list[int]:
        return self.labeling_strategy.get_all_labels()


class IdentityLabeler(Labeler):
    def __init__(
        self,
        labeling_strategy: LabelingStrategy,
        weighting_strategy: WeightingStrategy | None,
    ):
        self.labeling_strategy = labeling_strategy
        self.weighting_strategy = (
            weighting_strategy if weighting_strategy else NoWeighting()
        )

    def label_events(
        self, event_start_times: pd.DatetimeIndex, y: pd.Series
    ) -> EventDataFrame:
        forward_rolling_aggregated = y
        forward_rolling_aggregated = forward_rolling_aggregated[
            event_start_times
        ]  # filter based on events

        labels = self.labeling_strategy.label(forward_rolling_aggregated)
        sample_weights = self.weighting_strategy.calculate(forward_rolling_aggregated)

        end_times = y.index[
            (y.index.get_indexer(event_start_times) + 1).clip(max=len(y) - 1)
        ]
        return Artifact.events_from_data(
            start=event_start_times,
            end=end_times,
            label=labels,
            raw=forward_rolling_aggregated,
            sample_weights=sample_weights,
            test_sample_weights=sample_weights,
        )

    def get_all_possible_labels(self) -> list[int]:
        return self.labeling_strategy.get_all_labels()
