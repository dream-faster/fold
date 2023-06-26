import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fold.composites import Concat
from fold.composites.concat import Pipeline
from fold.composites.optimize import OptimizeGridSearch
from fold.composites.select import SelectBest
from fold.events import CreateEvents
from fold.events.filters.everynth import EveryNth
from fold.events.labeling import BinarizeSign, FixedForwardHorizon
from fold.events.weights import NoWeighing
from fold.loop import train, train_evaluate
from fold.loop.backtesting import backtest
from fold.loop.encase import train_backtest
from fold.models import WrapSKLearnClassifier
from fold.splitters import ExpandingWindowSplitter, SlidingWindowSplitter
from fold.transformations import AddWindowFeatures, RemoveLowVarianceFeatures
from fold.transformations.function import ApplyFunction
from fold.transformations.lags import AddLagsX, AddLagsY
from fold.utils.dataset import get_preprocessed_dataset


@pytest.mark.parametrize("backend", ["no", "ray", "thread"])
def test_on_weather_data_backends(backend: str) -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )
    events = FixedForwardHorizon(
        2, BinarizeSign(), weighing_strategy=NoWeighing()
    ).label_events(y.index, y)
    splitter = SlidingWindowSplitter(train_window=0.2, step=0.2)
    pipeline = [
        Concat(
            [
                Concat(
                    [
                        Pipeline(
                            AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))])
                        ),
                        AddLagsY(list(range(4, 7))),
                        AddWindowFeatures(("pressure", 14, "mean")),
                    ]
                ),
                AddWindowFeatures(("humidity", 26, "std")),
                Concat(
                    [
                        ApplyFunction(
                            lambda x: x.rolling(30).mean(), past_window_size=30
                        ),
                    ]
                ),
            ]
        ),
        RemoveLowVarianceFeatures(),
        HistGradientBoostingRegressor(),
    ]

    pred, _, insample_predictions = train_backtest(
        pipeline,
        X,
        y,
        splitter,
        events=events,
        backend=backend,
        return_insample=True,
    )
    assert len(pred) == 800
    assert (
        len(insample_predictions) <= 800
    ), "length of insample predictions should be always <= length of outofsample predictions"


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
        CreateEvents(
            RandomForestClassifier(),
            FixedForwardHorizon(
                1, labeling_strategy=BinarizeSign(), weighing_strategy=None
            ),
        ),
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
    pipeline = OptimizeGridSearch(
        [
            AddLagsY(list(range(1, 3))),
            AddLagsX(columns_and_lags=[("pressure", list(range(1, 10)))]),
            CreateEvents(
                SelectBest(
                    [
                        WrapSKLearnClassifier.from_model(LogisticRegression()),
                        WrapSKLearnClassifier.from_model(RandomForestClassifier()),
                    ]
                ),
                FixedForwardHorizon(
                    time_horizon=5,
                    labeling_strategy=BinarizeSign(),
                    weighing_strategy=None,
                ),
                EveryNth(2),
            ),
        ],
        krisi_metric_key="f_one_score_macro",
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
