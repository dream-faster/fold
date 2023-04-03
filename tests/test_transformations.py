import numpy as np
import pandas as pd
import pytest

from fold.composites.columns import PerColumnTransform
from fold.composites.concat import TransformColumn
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import DropColumns, SelectColumns
from fold.transformations.date import AddDateTimeFeatures, DateTimeFeature
from fold.transformations.dev import Identity
from fold.transformations.difference import Difference
from fold.transformations.holidays import AddHolidayFeatures, LabelingMethod
from fold.transformations.lags import AddLagsX, AddLagsY
from fold.transformations.window import AddWindowFeatures
from fold.utils.tests import generate_sine_wave_data


def test_no_transformation() -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [Identity()]

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()


def test_nested_transformations() -> None:
    X, y = generate_sine_wave_data()
    X["sine_2"] = X["sine"]

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        TransformColumn("sine_2", [lambda x: x + 2.0, lambda x: x + 1.0]),
        SelectColumns("sine_2"),
    ]

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (
        np.isclose((X["sine_2"][pred.index]).values, (pred.squeeze() - 3.0).values)
    ).all()


def test_column_select_single_column_transformation() -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [SelectColumns(columns=["sine_2"])]

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (X["sine_2"][pred.index] == pred.squeeze()).all()


def test_function_transformation() -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [lambda x: x - 1.0]

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
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

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (np.isclose((X.loc[pred.index].sum(axis=1) + 4.0), pred.squeeze())).all()


def test_add_lags_y():
    X, y = generate_sine_wave_data(length=6000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = AddLagsY(lags=[1, 2, 3])
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred["y_lag_1"] == y.shift(1)[pred.index]).all()
    assert (pred["y_lag_2"] == y.shift(2)[pred.index]).all()
    assert (pred["y_lag_3"] == y.shift(3)[pred.index]).all()

    transformations = AddLagsY(lags=range(1, 4))
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred["y_lag_1"] == y.shift(1)[pred.index]).all()
    assert (pred["y_lag_2"] == y.shift(2)[pred.index]).all()
    assert (pred["y_lag_3"] == y.shift(3)[pred.index]).all()

    with pytest.raises(ValueError, match="lags must be a range or a List"):
        transformations = AddLagsY(lags=0)  # type: ignore
        trained_pipelines = train(transformations, X, y, splitter)


def test_add_lags_X():
    X, y = generate_sine_wave_data(length=6000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = AddLagsX(columns_and_lags=[("sine", [1, 2, 3])])
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred["sine_lag_1"] == X["sine"].shift(1)[pred.index]).all()
    assert (pred["sine_lag_2"] == X["sine"].shift(2)[pred.index]).all()
    assert (pred["sine_lag_3"] == X["sine"].shift(3)[pred.index]).all()

    transformations = AddLagsX(columns_and_lags=[("sine", range(1, 4))])
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred["sine_lag_1"] == X["sine"].shift(1)[pred.index]).all()
    assert (pred["sine_lag_2"] == X["sine"].shift(2)[pred.index]).all()
    assert (pred["sine_lag_3"] == X["sine"].shift(3)[pred.index]).all()

    X["sine_inverted"] = generate_sine_wave_data(length=6000)[0].squeeze() * -1.0
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = AddLagsX(
        columns_and_lags=[("sine", [1, 2, 3]), ("all", [5, 8, 11])]
    )
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred["sine_lag_1"] == X["sine"].shift(1)[pred.index]).all()
    assert (pred["sine_lag_2"] == X["sine"].shift(2)[pred.index]).all()
    assert (pred["sine_lag_3"] == X["sine"].shift(3)[pred.index]).all()
    assert (pred["sine_lag_5"] == X["sine"].shift(5)[pred.index]).all()
    assert (pred["sine_lag_8"] == X["sine"].shift(8)[pred.index]).all()
    assert (pred["sine_lag_11"] == X["sine"].shift(11)[pred.index]).all()
    assert (
        pred["sine_inverted_lag_5"] == X["sine_inverted"].shift(5)[pred.index]
    ).all()
    assert (
        pred["sine_inverted_lag_8"] == X["sine_inverted"].shift(8)[pred.index]
    ).all()
    assert (
        pred["sine_inverted_lag_11"] == X["sine_inverted"].shift(11)[pred.index]
    ).all()
    assert len(pred.columns) == 11


def test_difference():
    X, y = generate_sine_wave_data(length=600)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = Difference()
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert np.isclose(
        X.squeeze()[pred.index],
        trained_pipelines[0].iloc[0].inverse_transform(pred).squeeze(),
        atol=1e-3,
    ).all()


def test_holiday_features_daily() -> None:
    X, y = generate_sine_wave_data()
    new_index = pd.date_range(start="1/1/2018", periods=len(X))
    X.index = new_index
    y.index = new_index

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    trained_pipelines = train(
        AddHolidayFeatures(["US", "DE"], labeling="holiday_binary"), X, y, splitter
    )
    pred = backtest(trained_pipelines, X, y, splitter)

    assert (np.isclose((X.squeeze()[pred.index]), (pred["sine"]))).all()
    assert (
        pred["US_holiday"]["2019-12-25"] == 1
    ), "Christmas should be a holiday for US."
    assert (
        pred["DE_holiday"]["2019-12-25"] == 1
    ), "Christmas should be a holiday for DE."
    assert pred["DE_holiday"]["2019-12-29"] == 0, "2019-12-29 is not a holiday in DE"


def test_holiday_features_minute() -> None:
    X, y = generate_sine_wave_data()
    new_index = pd.date_range(start="2021-12-06", freq="H", periods=len(X))
    X.index = new_index
    y.index = new_index

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    trained_pipelines = train(
        AddHolidayFeatures(["US", "DE"], labeling="holiday_binary"), X, y, splitter
    )
    pred = backtest(trained_pipelines, X, y, splitter)

    assert (np.isclose((X.squeeze()[pred.index]), (pred["sine"]))).all()
    assert (
        pred["US_holiday"]["2021-12-25"].mean() == 1
    ), "Christmas should be a holiday for US."
    assert (
        pred["DE_holiday"]["2021-12-25"].mean() == 1
    ), "Christmas should be a holiday for DE."

    trained_pipelines = train(
        AddHolidayFeatures(["DE"], labeling="weekday_weekend_holiday"), X, y, splitter
    )
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (
        pred["DE_holiday"]["2021-12-25"].mean() == 2
    ), "2021-12-25 should be both a holiday and a weekend (holiday taking precedence)."

    trained_pipelines = train(
        AddHolidayFeatures(
            ["US"],
            labeling=LabelingMethod.weekday_weekend_uniqueholiday,
            label_encode=False,
        ),
        X,
        y,
        splitter,
    )
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (
        pred["US_holiday"]["2021-12-25"].iloc[0] == "Christmas Day"
    ), "2021-12-25 should be a holiday string."

    trained_pipelines = train(
        AddHolidayFeatures(
            ["US", "DE"], labeling="weekday_weekend_uniqueholiday", label_encode=True
        ),
        X,
        y,
        splitter,
    )
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (
        pred["US_holiday"]["2021-12-31"].mean() == 14.0
    ), "2021-12-31 should be a holiday with a special id."


def test_datetime_features():
    X, y = generate_sine_wave_data(length=6000, freq="1min")
    splitter = ExpandingWindowSplitter(initial_train_window=0.5, step=0.15)
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
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
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


def test_window_features():
    X, y = generate_sine_wave_data(length=600)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = AddWindowFeatures(("sine", 14, "mean"))
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])

    # check if it works when passing a list of tuples
    transformations = AddWindowFeatures([("sine", 14, "mean")])
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])

    # check if it works with multiple transformations
    transformations = AddWindowFeatures([("sine", 14, "mean"), ("sine", 5, "max")])
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])
    assert pred["sine_5_max"].equals(X["sine"].rolling(5).max()[pred.index])

    transformations = AddWindowFeatures(
        [("sine", 14, lambda X: X.mean()), ("sine", 5, lambda X: X.max())]
    )
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    # if the Callable is lambda, then use the generic "transformed" name
    assert pred["sine_14_transformed"].equals(X["sine"].rolling(14).mean()[pred.index])
    assert pred["sine_5_transformed"].equals(X["sine"].rolling(5).max()[pred.index])

    transformations = AddWindowFeatures(
        [
            ("sine", 14, pd.core.window.rolling.Rolling.mean),
        ]
    )
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    # it should pick up the name of the function
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])

    X["sine_inverted"] = generate_sine_wave_data(length=6000)[0].squeeze() * -1.0
    transformations = AddWindowFeatures([("sine", 14, "mean"), ("all", 5, "mean")])
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    # it should pick up the name of the function
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])
    assert pred["sine_5_mean"].equals(X["sine"].rolling(5).mean()[pred.index])
    assert pred["sine_inverted_5_mean"].equals(
        X["sine_inverted"].rolling(5).mean()[pred.index]
    )
    assert len(pred.columns) == 5


def test_drop_columns():
    X, y = generate_sine_wave_data(length=600)
    X["sine_inverted"] = generate_sine_wave_data(length=6000)[0].squeeze() * -1.0
    X["sine_inverted_double"] = generate_sine_wave_data(length=6000)[0].squeeze() * -2.0
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = DropColumns(["sine_inverted", "sine"])
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert len(pred.columns) == 1

    transformations = DropColumns("all")
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert len(pred.columns) == 0
