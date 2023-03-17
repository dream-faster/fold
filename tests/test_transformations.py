import numpy as np

from fold.composites.columns import PerColumnTransform
from fold.composites.concat import TransformColumn
from fold.loop import backtest, train
from fold.loop.types import TrainMethod
from fold.models.base import Model
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import Identity, SelectColumns
from fold.transformations.difference import Difference
from fold.transformations.lags import AddLagsY
from fold.utils.tests import generate_sine_wave_data


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


def test_add_lags_y_minibatch_sequential() -> None:
    X, y = generate_sine_wave_data()
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = AddLagsY(lags=[1, 2, 3])
    transformations_over_time = train(
        transformations, X, y, splitter, train_method=TrainMethod.sequential
    )
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred["y_lag_1"] == y.shift(1)[pred.index]).all()
    assert (pred["y_lag_2"] == y.shift(2)[pred.index]).all()
    assert (pred["y_lag_3"] == y.shift(3)[pred.index]).all()


def test_add_lags_y_online():
    X, y = generate_sine_wave_data(resolution=600)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    transformations = AddLagsY(lags=[1, 2, 3])
    transformations.properties.mode = Model.Properties.Mode.online
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred["y_lag_1"] == y.shift(1)[pred.index]).all()
    assert (pred["y_lag_2"] == y.shift(2)[pred.index]).all()
    assert (pred["y_lag_3"] == y.shift(3)[pred.index]).all()


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
