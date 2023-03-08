import numpy as np
from sklearn.base import TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import (
    OnlyPredictions,
    RenameColumns,
    SelectColumns,
    TransformColumn,
)
from fold.utils.tests import generate_all_zeros, generate_sine_wave_data


def test_sklearn_classifier() -> None:
    X = generate_all_zeros(1000)
    y = X.squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        DummyClassifier(strategy="constant", constant=0),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()


def test_sklearn_regressor() -> None:
    X = generate_all_zeros(1000)
    y = X.squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        DummyRegressor(strategy="constant", constant=0),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()


def test_sklearn_pipeline() -> None:
    X = generate_all_zeros(1000)
    y = X.squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("dummy", DummyRegressor(strategy="constant", constant=0)),
            ]
        ),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()


def test_sklearn_partial_fit() -> None:
    X = generate_all_zeros(1000)
    y = X.squeeze()

    class TestEstimator(TransformerMixin):
        fit_called = False
        partial_fit_called = False

        def fit(self, X, y) -> None:
            self.fit_called = True

        def partial_fit(self, X, y) -> None:
            self.partial_fit_called = True

        def transform(self, X):
            return X

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        TestEstimator(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    _ = backtest(transformations_over_time, X, y, splitter)
    assert transformations_over_time[0].iloc[0].transformation.fit_called is False
    assert (
        transformations_over_time[0].iloc[0].transformation.partial_fit_called is True
    )


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
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (np.isclose((X["sine"][pred.index]), pred.squeeze())).all()
    assert pred.squeeze().name == "pred"
