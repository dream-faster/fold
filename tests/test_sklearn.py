import numpy as np
from sklearn.base import TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fold.composites.concat import TransformColumn
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions, RenameColumns, SelectColumns
from fold.utils.tests import generate_all_zeros, generate_sine_wave_data


def test_sklearn_classifier() -> None:
    X, y = generate_all_zeros(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        DummyClassifier(strategy="constant", constant=0),
        OnlyPredictions(),
    ]
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()


def test_sklearn_regressor() -> None:
    X, y = generate_all_zeros(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        DummyRegressor(strategy="constant", constant=0),
        OnlyPredictions(),
    ]
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()


def test_sklearn_pipeline() -> None:
    X, y = generate_all_zeros(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("dummy", DummyRegressor(strategy="constant", constant=0)),
            ]
        ),
        OnlyPredictions(),
    ]
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()


def test_sklearn_partial_fit() -> None:
    X, y = generate_all_zeros(1000)

    class TestEstimator(TransformerMixin):
        fit_called = False
        partial_fit_called = False

        def fit(self, X, y) -> None:
            self.fit_called = True

        def partial_fit(self, X, y) -> None:
            self.partial_fit_called = True

        def transform(self, X):
            return X

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        TestEstimator(),
    ]
    trained_pipelines = train(transformations, X, y, splitter)
    _ = backtest(trained_pipelines, X, y, splitter)
    assert trained_pipelines[0].iloc[0].transformation.fit_called is False
    assert trained_pipelines[0].iloc[0].transformation.partial_fit_called is True


def test_nested_transformations_with_feature_selection() -> None:
    X, y = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1
    X["constant"] = 1.0
    X["constant2"] = 2.0

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        VarianceThreshold(),
        TransformColumn("sine_2", [lambda x: x**2.0]),
        SelectKBest(score_func=f_regression, k=1),
        RenameColumns({"sine": "pred"}),
        SelectColumns("pred"),
    ]

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (np.isclose((X["sine"][pred.index]), pred.squeeze())).all()
    assert pred.squeeze().name == "pred"
