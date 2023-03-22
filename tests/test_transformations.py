import numpy as np
import pandas as pd

from fold.composites.columns import PerColumnTransform
from fold.composites.concat import TransformColumn
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import AddHolidayFeatures
from fold.transformations.columns import SelectColumns
from fold.transformations.date import AddDateTimeFeatures, DateTimeFeature
from fold.transformations.dev import Identity
from fold.transformations.difference import Difference
from fold.transformations.lags import AddLagsX, AddLagsY
from fold.utils.tests import generate_all_zeros, generate_sine_wave_data


def test_no_transformation() -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [Identity()]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()


def test_nested_transformations() -> None:
    X, y = generate_sine_wave_data()
    X["sine_2"] = X["sine"]

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        TransformColumn("sine_2", [lambda x: x + 2.0, lambda x: x + 1.0]),
        SelectColumns("sine_2"),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (
        np.isclose((X["sine_2"][pred.index]).values, (pred.squeeze() - 3.0).values)
    ).all()


def test_column_select_single_column_transformation() -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [SelectColumns(columns=["sine_2"])]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (X["sine_2"][pred.index] == pred.squeeze()).all()


def test_function_transformation() -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [lambda x: x - 1.0]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (np.isclose((X.squeeze()[pred.index]), (pred.squeeze() + 1.0))).all()


def test_per_column_transform() -> None:
    X, y = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1.0
    X["sine_3"] = X["sine"] + 2.0
    X["sine_4"] = X["sine"] + 3.0

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        PerColumnTransform([lambda x: x, lambda x: x + 1.0]),
        lambda x: x.sum(axis=1).to_frame(),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (np.isclose((X.loc[pred.index].sum(axis=1) + 4.0), pred.squeeze())).all()


def test_add_lags_y():
    X, y = generate_sine_wave_data(resolution=6000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = AddLagsY(lags=[1, 2, 3])
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred["y_lag_1"] == y.shift(1)[pred.index]).all()
    assert (pred["y_lag_2"] == y.shift(2)[pred.index]).all()
    assert (pred["y_lag_3"] == y.shift(3)[pred.index]).all()


def test_add_lags_X():
    X, y = generate_sine_wave_data(resolution=6000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = AddLagsX(columns_and_lags=[("sine", [1, 2, 3])])
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred["sine_lag_1"] == X["sine"].shift(1)[pred.index]).all()
    assert (pred["sine_lag_2"] == X["sine"].shift(2)[pred.index]).all()
    assert (pred["sine_lag_3"] == X["sine"].shift(3)[pred.index]).all()


def test_difference():
    X, y = generate_sine_wave_data(resolution=600)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = Difference()
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(
        X.squeeze()[pred.index],
        transformations_over_time[0].iloc[0].inverse_transform(pred).squeeze(),
        atol=1e-3,
    ).all()


def test_holiday_daily_transformation() -> None:
    X, y = generate_sine_wave_data()
    new_index = pd.date_range(start="1/1/2018", periods=len(X))
    X.index = new_index
    y.index = new_index

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [AddHolidayFeatures(["US", "DE"], type="holiday_binary")]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)

    assert (np.isclose((X.squeeze()[pred.index]), (pred["sine"]))).all()
    assert (
        pred["US_holiday_binary"]["2019-12-25"] == 1.0
    ), "Christmas should be a holiday for US."
    assert (
        pred["DE_holiday_binary"]["2019-12-25"] == 1.0
    ), "Christmas should be a holiday for DE."
    assert (
        pred["DE_holiday_binary"]["2019-12-29"] == 0.0
    ), "2019-12-29 is not a holiday in DE"


def test_holiday_minute_transformation() -> None:
    X, y = generate_sine_wave_data()
    new_index = pd.date_range(start="2021-12-25", freq="H", periods=len(X))
    X.index = new_index
    y.index = new_index

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        AddHolidayFeatures(["US", "DE"], type="holiday_binary"),
        AddHolidayFeatures(["DE"], type="weekday_weekend_holiday"),
        AddHolidayFeatures(["US"], type="weekday_weekend_uniqueholiday"),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)

    assert (np.isclose((X.squeeze()[pred.index]), (pred["sine"]))).all()
    assert (
        pred["US_holiday_binary"]["2021-12-25"].mean() == 1.0
    ), "Christmas should be a holiday for US."
    assert (
        pred["DE_holiday_binary"]["2021-12-25"].mean() == 1.0
    ), "Christmas should be a holiday for DE."
    assert (
        pred["DE_weekday_weekend_holiday"]["2021-12-25"].mean() == 3.0
    ), "2021-12-25 should be both a holiday and a weekend."
    assert (
        pred["US_weekday_weekend_uniqueholiday"]["2021-12-31"].mean() == 12.0
    ), "2021-12-31 should be a holiday with a special id."


def test_datetime_features():
    X, y = generate_all_zeros(length=600)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = AddDateTimeFeatures(
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
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred["second"] == X.loc[pred.index].index.second).all()
    assert (pred["minute"] == X.loc[pred.index].index.minute).all()
    assert (pred["hour"] == X.loc[pred.index].index.hour).all()
    assert (pred["day_of_week"] == X.loc[pred.index].index.dayofweek).all()
    assert (pred["day_of_month"] == X.loc[pred.index].index.day).all()
    assert (pred["day_of_year"] == X.loc[pred.index].index.dayofyear).all()
    assert (pred["week"] == X.loc[pred.index].index.week).all()
    assert (pred["week_of_year"] == X.loc[pred.index].index.weekofyear).all()
    assert (pred["month"] == X.loc[pred.index].index.month).all()
    assert (pred["quarter"] == X.loc[pred.index].index.quarter).all()
    assert (pred["year"] == X.loc[pred.index].index.year).all()
