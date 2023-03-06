from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.utils.tests import generate_all_zeros


def test_sklearn_classifier() -> None:

    X = generate_all_zeros(1000)
    y = X.squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        DummyClassifier(strategy="constant", constant=0),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
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
    _, pred = backtest(transformations_over_time, X, y, splitter)
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
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()
