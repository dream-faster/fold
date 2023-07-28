# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional, Union

import pandas as pd

from ..base import SingleFunctionTransformation, Transformation, Tunable, fit_noop
from ..utils.enums import ParsableEnum
from ..utils.list import wrap_in_list


class DateTimeFeature(ParsableEnum):
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


class AddDateTimeFeatures(Transformation, Tunable):
    """
    Adds (potentially multiple) date/time features to the input, as additional columns.
    The name of the new column will be the name of the DateTimeFeature passed in.
    Values are returned as integers, so the 59th minute of the hour will be `59`, and year 2022 will be `2022`.

    Parameters
    ----------

    features: List[DateTimeFeature, str]
        The features to add to the input. Options: `second`, `minute`, `hour`, `day_of_week`, `day_of_month`, `day_of_year`, `week`, `week_of_year`, `month`, `quarter`, `year`.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import AddDateTimeFeatures
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data(freq="min")
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = AddDateTimeFeatures(["minute"])
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                           sine  minute
    2021-12-31 15:40:00 -0.0000      40
    2021-12-31 15:41:00  0.0126      41
    2021-12-31 15:42:00  0.0251      42
    2021-12-31 15:43:00  0.0377      43
    2021-12-31 15:44:00  0.0502      44

    ```
    """

    def __init__(
        self,
        features: List[Union[DateTimeFeature, str]],
        keep_original: bool = True,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.features = [DateTimeFeature(f) for f in wrap_in_list(features)]
        self.name = (
            name
            if name is not None
            else f"AddDateTimeFeatures-{'-'.join([i.value for i in self.features])}"
        )
        self.keep_original = keep_original
        self.params_to_try = params_to_try
        self.properties = Transformation.Properties(requires_X=False)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        X_datetime = pd.DataFrame([], index=X.index)
        for feature in self.features:
            if feature == DateTimeFeature.second:
                X_datetime[feature.value] = X.index.second
            elif feature == DateTimeFeature.minute:
                X_datetime[feature.value] = X.index.minute
            elif feature == DateTimeFeature.hour:
                X_datetime[feature.value] = X.index.hour
            elif feature == DateTimeFeature.day_of_week:
                X_datetime[feature.value] = X.index.dayofweek
            elif feature == DateTimeFeature.day_of_month:
                X_datetime[feature.value] = X.index.day
            elif feature == DateTimeFeature.day_of_year:
                X_datetime[feature.value] = X.index.dayofyear
            elif (
                feature == DateTimeFeature.week
                or feature == DateTimeFeature.week_of_year
            ):
                X_datetime[feature.value] = pd.Index(
                    X.index.isocalendar().week, dtype="int"
                )
            elif feature == DateTimeFeature.month:
                X_datetime[feature.value] = X.index.month
            elif feature == DateTimeFeature.quarter:
                X_datetime[feature.value] = X.index.quarter
            elif feature == DateTimeFeature.year:
                X_datetime[feature.value] = X.index.year
            else:
                raise ValueError(f"Unsupported feature: {feature}")
        to_concat = [X, X_datetime] if self.keep_original else X_datetime
        concatenated = pd.concat(to_concat, copy=False, axis="columns")
        return concatenated

    fit = fit_noop
    update = fit_noop


class AddSecond(SingleFunctionTransformation):
    """
    Adds "second" features to the input, as an additional column.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import AddSecond
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data(freq="S")
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = AddSecond()
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                           sine  second
    2021-12-31 23:51:40 -0.0000      40
    2021-12-31 23:51:41  0.0126      41
    2021-12-31 23:51:42  0.0251      42
    2021-12-31 23:51:43  0.0377      43
    2021-12-31 23:51:44  0.0502      44

    ```
    """

    name = "AddSecond"

    def get_function(self) -> Callable:
        return lambda X: X.index.second.to_series().rename("second").set_axis(X.index)


class AddMinute(SingleFunctionTransformation):
    """
    Adds "minute" features to the input, as an additional column.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import AddMinute
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data(freq="min")
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = AddMinute()
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                           sine  minute
    2021-12-31 15:40:00 -0.0000      40
    2021-12-31 15:41:00  0.0126      41
    2021-12-31 15:42:00  0.0251      42
    2021-12-31 15:43:00  0.0377      43
    2021-12-31 15:44:00  0.0502      44

    ```
    """

    name = "AddMinute"

    def get_function(self) -> Callable:
        return lambda X: X.index.minute.to_series().rename("minute").set_axis(X.index)


class AddHour(SingleFunctionTransformation):
    name = "AddHour"

    def get_function(self) -> Callable:
        return lambda X: X.index.hour.to_series().rename("hour").set_axis(X.index)


class AddDayOfWeek(SingleFunctionTransformation):
    name = "AddDayOfWeek"

    def get_function(self) -> Callable:
        return (
            lambda X: X.index.dayofweek.to_series()
            .rename("day_of_week")
            .set_axis(X.index)
        )


class AddDayOfMonth(SingleFunctionTransformation):
    name = "AddDayOfMonth"

    def get_function(self) -> Callable:
        return (
            lambda X: X.index.day.to_series().rename("day_of_month").set_axis(X.index)
        )


class AddDayOfYear(SingleFunctionTransformation):
    name = "AddDayOfYear"

    def get_function(self) -> Callable:
        return (
            lambda X: X.index.dayofyear.to_series()
            .rename("day_of_year")
            .set_axis(X.index)
        )


class AddWeek(SingleFunctionTransformation):
    name = "AddWeek"

    def get_function(self) -> Callable:
        return (
            lambda X: pd.Index(X.index.isocalendar().week, dtype="int")
            .to_series()
            .set_axis(X.index)
            .rename("week")
        )


class AddWeekOfYear(SingleFunctionTransformation):
    name = "AddWeekOfYear"

    def get_function(self) -> Callable:
        return (
            lambda X: pd.Index(X.index.isocalendar().week, dtype="int")
            .to_series()
            .set_axis(X.index)
            .rename("week_of_year")
        )


class AddMonth(SingleFunctionTransformation):
    name = "AddMonth"

    def get_function(self) -> Callable:
        return lambda X: X.index.month.to_series().rename("month").set_axis(X.index)


class AddQuarter(SingleFunctionTransformation):
    name = "AddQuarter"

    def get_function(self) -> Callable:
        return lambda X: X.index.quarter.to_series().rename("quarter").set_axis(X.index)


class AddYear(SingleFunctionTransformation):
    name = "AddYear"

    def get_function(self) -> Callable:
        return lambda X: X.index.year.to_series().rename("year").set_axis(X.index)
