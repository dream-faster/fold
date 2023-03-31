from __future__ import annotations

from enum import Enum
from typing import List, Union

import pandas as pd

from .base import Transformation, fit_noop


class DateTimeFeature(Enum):
    second = "second"
    minute = "minute"
    hour = "hour"
    day_of_week = "day_of_week"
    day_of_month = "day_of_month"
    day_of_year = "day_of_year"
    week = "week"
    week_of_year = "week_of_year"
    month = "month"
    quarter = "quarter"
    year = "year"

    @staticmethod
    def from_str(value: Union[str, DateTimeFeature]) -> DateTimeFeature:
        if isinstance(value, DateTimeFeature):
            return value
        for strategy in DateTimeFeature:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown DateTimeFeature: {value}")


class AddDateTimeFeatures(Transformation):
    """
    Adds (potentially multiple) date/time features to the input, as additional columns.
    The name of the new column will be the name of the DateTimeFeature passed in.
    Currently, we don't encode the values.

    Parameters
    ----------

    features: List[Union[DateTimeFeature, str]]
        The features to add to the input.

    """

    properties = Transformation.Properties(requires_X=False)
    name = "AddDateTimeFeatures"

    def __init__(
        self,
        features: List[Union[DateTimeFeature, str]],
    ) -> None:
        self.features = [DateTimeFeature(f) for f in features]

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        X = X.copy()
        for feature in self.features:
            if feature == DateTimeFeature.second:
                X[feature.value] = X.index.second
            elif feature == DateTimeFeature.minute:
                X[feature.value] = X.index.minute
            elif feature == DateTimeFeature.hour:
                X[feature.value] = X.index.hour
            elif feature == DateTimeFeature.day_of_week:
                X[feature.value] = X.index.dayofweek
            elif feature == DateTimeFeature.day_of_month:
                X[feature.value] = X.index.day
            elif feature == DateTimeFeature.day_of_year:
                X[feature.value] = X.index.dayofyear
            elif feature == DateTimeFeature.week:
                X[feature.value] = X.index.week
            elif feature == DateTimeFeature.week_of_year:
                X[feature.value] = X.index.weekofyear
            elif feature == DateTimeFeature.month:
                X[feature.value] = X.index.month
            elif feature == DateTimeFeature.quarter:
                X[feature.value] = X.index.quarter
            elif feature == DateTimeFeature.year:
                X[feature.value] = X.index.year
            else:
                raise ValueError(f"Unsupported feature: {feature}")
        return X

    fit = fit_noop
    update = fit_noop
