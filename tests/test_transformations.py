import numpy as np

from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import Identity, SelectColumns
from fold.transformations.columns import PerColumnTransform, TransformColumn
from fold.transformations.lags import AddLagsY
from fold.transformations.target import TransformTarget
from fold.transformations.test import Test
from fold.utils.tests import generate_sine_wave_data, generate_zeros_and_ones_skewed


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


test_transform_plus_2 = Test(lambda x: x, lambda x: x + 2.0, lambda x: x - 2.0)


def all_y_values_above_1(X, y):
    assert np.all(y >= 1.0)


def all_y_values_below_1(X, y):
    assert np.all(y <= 1.0)


test_all_y_values_below_1 = Test(
    fit_func=all_y_values_below_1, transform_func=lambda X: X
)
test_all_y_values_above_1 = Test(
    fit_func=all_y_values_above_1, transform_func=lambda X: X
)


def test_target_transformation() -> None:
    X, y = generate_zeros_and_ones_skewed(length=1000, weights=[0.5, 0.5])

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)

    transformations = [
        TransformTarget(
            [lambda x: x + 1, test_all_y_values_above_1], test_transform_plus_2
        ),
        test_all_y_values_below_1,
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert ((X.squeeze()[pred.index] + 1) == pred.squeeze()).all()


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


def test_add_lags_y() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        AddLagsY(lags=[1, 2, 3]),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert pred["y_lag_1"].shift(1).equals(pred["y"])
    assert pred["y_lag_2"].shift(2).equals(pred["y"])
    assert pred["y_lag_3"].shift(3).equals(pred["y"])
