# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

import pandas as pd
from finml_utils.dataframes import concat_on_columns

from ..base import Transformation, Tunable, fit_noop
from .types import DateTimeFeature


class AddDateTimeFeatures(Transformation, Tunable):
    """
    Adds (potentially multiple) date/time features to the input, as additional columns.
    The name of the new column will be the name of the DateTimeFeature passed in.
    Values are returned as integers, so the 59th minute of the hour will be `59`, and year 2022 will be `2022`.

    Parameters
    ----------

    features: list[DateTimeFeature, str]
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
    >>> preds, trained_pipeline, _, _ = train_backtest(pipeline, X, y, splitter)
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
        features: list[DateTimeFeature | str],
        keep_original: bool = True,
        name: str | None = None,
        params_to_try: dict | None = None,
    ) -> None:
        self.features = [DateTimeFeature.from_str(f) for f in features]
        self.name = (
            name
            if name is not None
            else f"AddDateTimeFeatures-{'-'.join([i.value for i in self.features])}"
        )
        self.keep_original = keep_original
        self.params_to_try = params_to_try
        self.properties = Transformation.Properties(requires_X=False)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        datetime_features = []
        for feature in self.features:
            if feature is DateTimeFeature.second:
                datetime_features.append(
                    X.index.second.to_series().rename(feature.value).set_axis(X.index)
                )
            elif feature is DateTimeFeature.minute:
                datetime_features.append(
                    X.index.minute.to_series().rename(feature.value).set_axis(X.index)
                )
            elif feature == DateTimeFeature.hour:
                datetime_features.append(
                    X.index.hour.to_series().rename(feature.value).set_axis(X.index)
                )
            elif feature == DateTimeFeature.day_of_week:
                datetime_features.append(
                    X.index.dayofweek.to_series()
                    .rename(feature.value)
                    .set_axis(X.index)
                )
            elif feature == DateTimeFeature.day_of_month:
                datetime_features.append(
                    X.index.day.to_series().rename(feature.value).set_axis(X.index)
                )
            elif feature == DateTimeFeature.day_of_year:
                datetime_features.append(
                    X.index.dayofyear.to_series()
                    .rename(feature.value)
                    .set_axis(X.index)
                )
            elif feature in (DateTimeFeature.week, DateTimeFeature.week_of_year):
                datetime_features.append(
                    pd.Index(X.index.isocalendar().week, dtype="int")
                    .to_series()
                    .rename(feature.value)
                    .set_axis(X.index)
                )
            elif feature == DateTimeFeature.month:
                datetime_features.append(
                    X.index.month.to_series().rename(feature.value).set_axis(X.index)
                )
            elif feature == DateTimeFeature.quarter:
                datetime_features.append(
                    X.index.quarter.to_series().rename(feature.value).set_axis(X.index)
                )
            elif feature == DateTimeFeature.year:
                datetime_features.append(
                    X.index.year.to_series().rename(feature.value).set_axis(X.index)
                )
            else:
                raise ValueError(f"Unsupported feature: {feature}")
        datetime_features = concat_on_columns(datetime_features).add_prefix("datetime~")
        return (
            concat_on_columns(
                [X, datetime_features],
            )
            if self.keep_original
            else datetime_features
        )

    fit = fit_noop
    update = fit_noop
