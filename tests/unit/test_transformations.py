import numpy as np
import pandas as pd
import pytest

from fold.composites.columns import PerColumnTransform
from fold.composites.concat import TransformColumn
from fold.loop import backtest, train
from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from fold.transformations.columns import DropColumns, SelectColumns
from fold.transformations.dev import Identity
from fold.transformations.difference import Difference
from fold.transformations.lags import AddLagsX, AddLagsY
from fold.transformations.math import AddConstant, TakeLog, TurnPositive
from fold.transformations.window import AddWindowFeatures
from fold.utils.tests import generate_sine_wave_data


def test_no_transformation() -> None:
    X, y = generate_sine_wave_data()
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pred, _ = train_backtest([Identity()], X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()


def test_nested_transformations() -> None:
    X, y = generate_sine_wave_data()
    X["sine_2"] = X["sine"]

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        TransformColumn("sine_2", [lambda x: x + 2.0, lambda x: x + 1.0]),
        SelectColumns("sine_2"),
    ]

    pred, _ = train_backtest(transformations, X, y, splitter)
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
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (pred["y_lag_1"] == y.shift(1)[pred.index]).all()
    assert (pred["y_lag_2"] == y.shift(2)[pred.index]).all()
    assert (pred["y_lag_3"] == y.shift(3)[pred.index]).all()

    transformations = AddLagsY(lags=range(1, 4))
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (pred["y_lag_1"] == y.shift(1)[pred.index]).all()
    assert (pred["y_lag_2"] == y.shift(2)[pred.index]).all()
    assert (pred["y_lag_3"] == y.shift(3)[pred.index]).all()

    with pytest.raises(ValueError, match="lags must be a range or a List"):
        transformations = AddLagsY(lags=0)  # type: ignore
        _ = train(transformations, X, y, splitter)


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
    pred, trained_pipelines = train_backtest(transformations, X, y, splitter)
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


def test_window_features():
    X, y = generate_sine_wave_data(length=600)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = AddWindowFeatures(("sine", 14, "mean"))
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])

    # check if it works when passing a list of tuples
    transformations = AddWindowFeatures([("sine", 14, "mean")])
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])

    # check if it works with multiple transformations
    transformations = AddWindowFeatures([("sine", 14, "mean"), ("sine", 5, "max")])
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])
    assert pred["sine_5_max"].equals(X["sine"].rolling(5).max()[pred.index])

    transformations = AddWindowFeatures(
        [("sine", 14, lambda X: X.mean()), ("sine", 5, lambda X: X.max())]
    )
    pred, _ = train_backtest(transformations, X, y, splitter)
    # if the Callable is lambda, then use the generic "transformed" name
    assert pred["sine_14_transformed"].equals(X["sine"].rolling(14).mean()[pred.index])
    assert pred["sine_5_transformed"].equals(X["sine"].rolling(5).max()[pred.index])

    transformations = AddWindowFeatures(
        [
            ("sine", 14, pd.core.window.rolling.Rolling.mean),
        ]
    )
    pred, _ = train_backtest(transformations, X, y, splitter)
    # it should pick up the name of the function
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])

    X["sine_inverted"] = generate_sine_wave_data(length=6000)[0].squeeze() * -1.0
    transformations = AddWindowFeatures([("sine", 14, "mean"), ("all", 5, "mean")])
    pred, _ = train_backtest(transformations, X, y, splitter)
    # it should pick up the name of the function
    assert pred["sine_14_mean"].equals(X["sine"].rolling(14).mean()[pred.index])
    assert pred["sine_5_mean"].equals(X["sine"].rolling(5).mean()[pred.index])
    assert pred["sine_inverted_5_mean"].equals(
        X["sine_inverted"].rolling(5).mean()[pred.index]
    )
    assert len(pred.columns) == 5


def test_drop_columns():
    X, y = generate_sine_wave_data(length=600)
    X["sine_inverted"] = X["sine"] * -1.0
    X["sine_inverted_double"] = X["sine"] * -2.0
    splitter = SingleWindowSplitter(train_window=400)
    pred, _ = train_backtest(DropColumns(["sine_inverted", "sine"]), X, y, splitter)
    assert len(pred.columns) == 1
    assert pred["sine_inverted_double"].equals(X["sine_inverted_double"][pred.index])

    transformations = DropColumns("all")
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert len(pred.columns) == 0


def test_log_transformation():
    X, y = generate_sine_wave_data(length=600)
    X, y = X + 2.0, y + 2.0
    splitter = SingleWindowSplitter(train_window=400)
    pred, _ = train_backtest(TakeLog(), X, y, splitter)
    assert pred["sine"].equals(np.log(X["sine"][pred.index]))

    pred = TakeLog().inverse_transform(np.log(X["sine"]))
    assert np.isclose(pred, X["sine"][pred.index], atol=0.01).all()

    log = TakeLog(base=10)
    pred, _ = train_backtest(log, X, y, splitter)
    assert pred["sine"].equals(np.log10(X["sine"][pred.index]))

    pred = log.inverse_transform(np.log10(X["sine"]))
    assert np.isclose(pred, X["sine"][pred.index], atol=0.01).all()


def test_turn_positive():
    X, y = generate_sine_wave_data(length=600)
    X, y = X - 2.0, y - 2.0
    X["sine_inverted"] = X["sine"] * -1.0
    X["sine_inverted_double"] = X["sine"] * -2.0

    turn_positive = TurnPositive()
    turn_positive.fit(X, y, None)
    pred = turn_positive.transform(X, False)
    assert pred.any().any() >= 0.0
    assert len(pred.columns) == len(X.columns)

    reverse = turn_positive.inverse_transform(pred["sine"])
    assert np.isclose(reverse, X["sine"][pred.index], atol=0.01).all()


def test_add_constant():
    X, y = generate_sine_wave_data(length=600)
    X["sine_inverted"] = X["sine"] * -1.0
    X["sine_inverted_double"] = X["sine"] * -2.0
    splitter = SingleWindowSplitter(train_window=400)
    pred, _ = train_backtest(AddConstant(2.0), X, y, splitter)
    assert pred["sine"].equals(X["sine"][pred.index] + 2.0)

    pred = AddConstant(2.0).inverse_transform(X["sine"] + 2.0)
    assert np.isclose(pred, X["sine"][pred.index], atol=0.01).all()

    pred, _ = train_backtest(
        AddConstant({"sine": 2.0, "sine_inverted": 3.0}), X, y, splitter
    )
    assert pred["sine"].equals(X["sine"][pred.index] + 2.0)
    assert pred["sine_inverted"].equals(X["sine_inverted"][pred.index] + 3.0)
    assert len(pred.columns) == 3