import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier

from fold.composites import Concat
from fold.events import CreateEvents
from fold.events.filters.everynth import EveryNth
from fold.events.labeling import BinarizeSign, FixedForwardHorizon
from fold.loop import train, train_evaluate
from fold.loop.backtesting import backtest
from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import AddWindowFeatures
from fold.transformations.lags import AddLagsX, AddLagsY
from fold.utils.dataset import get_preprocessed_dataset


@pytest.mark.parametrize("backend", ["no", "ray", "thread"])
def test_on_weather_data_backends(backend: str) -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    pipeline = [
        Concat(
            [
                AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))]),
                AddLagsY(list(range(1, 10))),
                AddWindowFeatures(("pressure", 14, "mean")),
            ]
        ),
        HistGradientBoostingRegressor(),
    ]

    pred, _ = train_backtest(pipeline, X, y, splitter, backend=backend)
    assert len(pred) == 800


def test_train_evaluate() -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    pipeline = [
        AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))]),
        AddLagsY(list(range(1, 10))),
        HistGradientBoostingRegressor(),
    ]

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    scorecard, pred, trained_trained_pipelines = train_evaluate(
        pipeline, X, y, splitter
    )


def test_train_evaluate_probabilities() -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )
    y = y.pct_change()

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)

    pipeline = [
        AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))]),
        AddLagsY(list(range(1, 3))),
        CreateEvents(RandomForestClassifier(), FixedForwardHorizon(1, BinarizeSign())),
    ]

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    scorecard, pred, trained_trained_pipelines = train_evaluate(
        pipeline, X, y, splitter
    )


def test_integration_events() -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=500,
    )
    y = y.pct_change()

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    pipeline = CreateEvents(
        [
            # AddLagsX(columns_and_lags=[("pressure", list(range(1, 10)))]),
            AddLagsY(list(range(1, 3))),
            RandomForestClassifier(),
        ],
        FixedForwardHorizon(time_horizon=5, strategy=BinarizeSign()),
        EveryNth(2),
    )
    trained_pipeline = train(
        pipeline,
        X,
        y,
        splitter,
    )
    pred, artifacts = backtest(trained_pipeline, X, y, splitter, return_artifacts=True)
    # assert len(artifacts["label"]) == 184
    # assert len(pred) == 400
    # assert len(pred.dropna()) == 200
