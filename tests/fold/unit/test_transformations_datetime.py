import numpy as np
import pandas as pd

from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.date import AddDateTimeFeatures, DateTimeFeature
from fold.transformations.holidays import (  # AddExchangeHolidayFeatures,
    AddHolidayFeatures,
    LabelingMethod,
)
from fold.utils.tests import generate_sine_wave_data, tuneability_test


def test_datetime_features():
    X, y = generate_sine_wave_data(length=6000, freq="1min")
    splitter = ExpandingWindowSplitter(initial_train_window=0.5, step=0.15)
    pipeline = AddDateTimeFeatures(
        [
            DateTimeFeature.second,
            DateTimeFeature.minute,
            DateTimeFeature.hour,
            DateTimeFeature.day_of_week,
            DateTimeFeature.day_of_month,
            DateTimeFeature.day_of_year,
            DateTimeFeature.week,
            DateTimeFeature.week_of_year,
            DateTimeFeature.month,
            DateTimeFeature.quarter,
            DateTimeFeature.year,
        ]
    )
    pred, _, _, _ = train_backtest(pipeline, X, y, splitter)
    assert (pred["datetime~second"] == X.loc[pred.index].index.second).all()
    assert (pred["datetime~minute"] == X.loc[pred.index].index.minute).all()
    assert (pred["datetime~hour"] == X.loc[pred.index].index.hour).all()
    assert (pred["datetime~day_of_week"] == X.loc[pred.index].index.dayofweek).all()
    assert (pred["datetime~day_of_month"] == X.loc[pred.index].index.day).all()
    assert (pred["datetime~day_of_year"] == X.loc[pred.index].index.dayofyear).all()
    assert (pred["datetime~week"] == X.loc[pred.index].index.isocalendar().week).all()
    assert (
        pred["datetime~week_of_year"]
        == pd.Index(X.loc[pred.index].index.isocalendar().week, dtype="int")
    ).all()
    assert (pred["datetime~month"] == X.loc[pred.index].index.month).all()
    assert (pred["datetime~quarter"] == X.loc[pred.index].index.quarter).all()
    assert (pred["datetime~year"] == X.loc[pred.index].index.year).all()

    trans = AddDateTimeFeatures([DateTimeFeature.second, DateTimeFeature.minute])
    tuneability_test(trans, dict(features=[DateTimeFeature.day_of_week]))


def test_holiday_features_daily() -> None:
    X, y = generate_sine_wave_data()
    new_index = pd.date_range(start="1/1/2018", periods=len(X))
    X.index = new_index
    y.index = new_index

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pred, _, _, _ = train_backtest(
        AddHolidayFeatures(["US", "DE"], labeling="holiday_binary"), X, y, splitter
    )

    assert np.allclose((X.squeeze()[pred.index]), (pred["sine"]))
    assert (
        pred["holiday_US"]["2019-12-25"] == 1
    ), "Christmas should be a holiday for US."
    assert (
        pred["holiday_DE"]["2019-12-25"] == 1
    ), "Christmas should be a holiday for DE."
    assert pred["holiday_DE"]["2019-12-29"] == 0, "2019-12-29 is not a holiday in DE"


def test_holiday_features_minute() -> None:
    X, y = generate_sine_wave_data()
    new_index = pd.date_range(start="2021-12-06", freq="H", periods=len(X))
    X.index = new_index
    y.index = new_index

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pred, _, _, _ = train_backtest(
        AddHolidayFeatures(["US", "DE"], labeling="holiday_binary"), X, y, splitter
    )

    assert np.allclose(X.squeeze()[pred.index], pred["sine"])
    assert (
        pred["holiday_US"]["2021-12-25"].mean() == 1
    ), "Christmas should be a holiday for US."
    assert (
        pred["holiday_DE"]["2021-12-25"].mean() == 1
    ), "Christmas should be a holiday for DE."
    assert pd.api.types.is_integer_dtype(pred["holiday_US"].dtype)
    assert pd.api.types.is_integer_dtype(pred["holiday_DE"].dtype)

    pred, _, _, _ = train_backtest(
        AddHolidayFeatures(["DE"], labeling="weekday_weekend_holiday"), X, y, splitter
    )
    assert (
        pred["holiday_DE"]["2021-12-25"].mean() == 2
    ), "2021-12-25 should be both a holiday and a weekend (holiday taking precedence)."
    assert pd.api.types.is_integer_dtype(pred["holiday_DE"].dtype)

    pred, _, _, _ = train_backtest(
        AddHolidayFeatures(
            ["US"],
            labeling=LabelingMethod.weekday_weekend_uniqueholiday_string,
        ),
        X,
        y,
        splitter,
    )
    assert (
        pred["holiday_US"]["2021-12-25"].iloc[0] == "Christmas Day"
    ), "2021-12-25 should be a holiday string."
    assert pred["holiday_US"].dtype == "object"

    pred, _, _, _ = train_backtest(
        AddHolidayFeatures(["US", "DE"], labeling="weekday_weekend_uniqueholiday"),
        X,
        y,
        splitter,
    )
    assert (
        pred["holiday_US"]["2021-12-31"].mean() == 14.0
    ), "2021-12-31 should be a holiday with a special id."
    assert pd.api.types.is_integer_dtype(pred["holiday_US"].dtype)
    assert pd.api.types.is_integer_dtype(pred["holiday_DE"].dtype)


# def test_holiday_features_exchanges() -> None:
#     X, y = generate_sine_wave_data()
#     new_index = pd.date_range(start="1/1/2018", periods=len(X))
#     X.index = new_index
#     y.index = new_index

#     splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
#     trained = train(AddExchangeHolidayFeatures(["NYSE"]), X, y, splitter)
#     pred = backtest(trained, X, y, splitter)

#     assert np.allclose((X.squeeze()[pred.index]), (pred["sine"]))
#     assert (
#         pred["holiday_NYSE"]["2019-12-25"] == 2
#     ), "Christmas should be a holiday for NYSE."
