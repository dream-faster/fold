import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression

from drift.loop import backtest, train
from drift.splitters import ExpandingWindowSplitter
from drift.transformations import Identity, SelectColumns
from drift.transformations.base import Transformation
from drift.transformations.columns import (
    PerColumnTransform,
    RenameColumns,
    TransformColumn,
)
from drift.transformations.target import TransformTarget
from drift.utils.tests import generate_sine_wave_data


def test_no_transformation() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X["sine"].shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [Identity()]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred).all()


def test_nested_transformations() -> None:
    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"]
    y = X["sine"].shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        TransformColumn("sine_2", [lambda x: x + 2.0, lambda x: x + 1.0]),
        SelectColumns("sine_2"),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert np.all(np.isclose((X["sine_2"][pred.index]).values, (pred - 3.0).values))


def test_nested_transformations_with_feature_selection() -> None:
    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1
    X["constant"] = 1.0
    X["constant2"] = 2.0
    y = X["sine"].shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        VarianceThreshold(),
        TransformColumn("sine_2", [lambda x: x**2.0]),
        SelectKBest(score_func=f_regression, k=1),
        RenameColumns({"sine": "pred"}),
        SelectColumns("pred"),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert np.all(np.isclose((X["sine"][pred.index]).values, (pred).values))
    assert pred.name == "pred"


def test_column_select_single_column_transformation() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1
    y = X["sine"].shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [SelectColumns(columns=["sine_2"])]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (X["sine_2"][pred.index] == pred).all()


def test_function_transformation() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [lambda x: x - 1.0]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert np.all(np.isclose((X.squeeze()[pred.index]).values, (pred + 1.0).values))


class TestTransformTarget(Transformation):

    name = "test_transform_target"

    def fit(self, X, y):
        pass

    def transform(self, X):
        return X + 1.0

    def inverse_transform(self, X):
        return X - 1.0


class TestIfAllYValuesBelow1(Transformation):

    name = "TestIfAllYValuesBelow1"

    def fit(self, X, y):
        assert np.all(X <= 1.0)
        return X

    def transform(self, X):
        return X


def test_target_transformation() -> None:

    X = generate_sine_wave_data()
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        TransformTarget(lambda x: x + 1, TestTransformTarget()),
        TestIfAllYValuesBelow1(),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert np.all(np.isclose((X.squeeze()[pred.index]).values, pred.values))


def test_per_column_transform() -> None:

    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1.0
    X["sine_3"] = X["sine"] + 2.0
    X["sine_4"] = X["sine"] + 3.0
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        PerColumnTransform([lambda x: x + 1.0]),
        lambda x: x.sum(axis=1),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert np.all(np.isclose((X.loc[pred.index].sum(axis=1) + 4.0).values, pred.values))
