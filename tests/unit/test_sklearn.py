import numpy as np
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression

from fold.composites.concat import TransformColumn
from fold.loop import backtest, train, train_backtest
from fold.models.sklearn import WrapSKLearnClassifier, WrapSKLearnRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions, RenameColumns, SelectColumns
from fold.utils.tests import (
    generate_all_zeros,
    generate_sine_wave_data,
    tuneability_test,
)


def test_sklearn_classifier() -> None:
    X, y = generate_all_zeros(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [DummyClassifier(strategy="constant", constant=0), OnlyPredictions()]
    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()

    tuneability_test(
        instance=WrapSKLearnClassifier.from_model(pipeline[0]),
        different_params=dict(strategy="constant", constant=1),
        init_function=lambda **kwargs: WrapSKLearnClassifier.from_model(
            DummyClassifier(**kwargs)
        ),
        classification=True,
    )


def test_sklearn_regressor() -> None:
    X, y = generate_all_zeros(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        DummyRegressor(strategy="constant", constant=0),
        OnlyPredictions(),
    ]
    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()

    tuneability_test(
        instance=WrapSKLearnRegressor.from_model(pipeline[0]),
        different_params=dict(strategy="constant", constant=1),
        init_function=lambda **kwargs: WrapSKLearnRegressor.from_model(
            DummyRegressor(**kwargs)
        ),
    )


def test_sklearn_transformation_variable_columns() -> None:
    X, y = generate_sine_wave_data()
    X["sine1"] = X["sine"].shift(1)
    X["sine2"] = X["sine"].shift(2)
    X["sine3"] = X["sine"].shift(3)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        PCA(n_components=2),
    ]
    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert pred.shape[1] == 2


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

        def get_params(self, deep=True):
            return {}

        def clone_with_params(self, parameters: dict, clone_children=None):
            return self

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
