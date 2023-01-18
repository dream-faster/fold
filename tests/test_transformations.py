import numpy as np
import pytest

from drift.loop import infer, train
from drift.splitters import ExpandingWindowSplitter
from drift.transformations import Identity, SelectColumns
from drift.transformations.columns import TransformColumn
from tests.utils import generate_sine_wave_data


def test_no_transformation() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [Identity()]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert (X.squeeze()[pred.index] == pred).all()


def test_nested_transformations() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        TransformColumn("sine_2", [lambda x: x + 1, lambda x: x + 1.0]),
        SelectColumns("sine_2"),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert np.all(np.isclose((X["sine_2"][pred.index]).values, (pred - 2.0).values))


def test_column_select_single_column_transformation() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [SelectColumns(columns=["sine_2"])]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert (X["sine_2"][pred.index] == pred).all()


def test_function_transformation() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X.shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [lambda x: x - 1.0]

    transformations_over_time = train(transformations, X.copy(), y, splitter)
    _, pred = infer(transformations_over_time, X.copy(), splitter)
    assert np.all(np.isclose((X.squeeze()[pred.index]).values, (pred + 1.0).values))
