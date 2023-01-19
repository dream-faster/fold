import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import LinearRegression

from drift.loop import infer, train
from drift.models import Ensemble
from drift.splitters import ExpandingWindowSplitter
from drift.transformations import Identity, SelectColumns
from drift.transformations.columns import RenameColumns, TransformColumn
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
    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"]
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        TransformColumn("sine_2", [lambda x: x + 2.0, lambda x: x + 1.0]),
        SelectColumns("sine_2"),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert np.all(np.isclose((X["sine_2"][pred.index]).values, (pred - 3.0).values))


def test_nested_transformations_with_feature_selection() -> None:
    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1
    X["constant"] = 1.0
    X["constant2"] = 2.0
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        VarianceThreshold(),
        TransformColumn("sine_2", [lambda x: x + 1.0]),
        SelectKBest(score_func=f_regression, k=1),
        RenameColumns({"sine_2": "pred"}),
        SelectColumns("pred"),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert np.all(np.isclose((X["sine_2"][pred.index]).values, (pred - 1.0).values))
    assert pred.name == "pred"


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
