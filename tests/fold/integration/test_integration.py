import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from fold.base.classes import PipelineCard
from fold.composites import Concat
from fold.composites.concat import Sequence
from fold.composites.optimize import OptimizeGridSearch
from fold.composites.select import SelectBest
from fold.events.filters.everynth import EveryNth
from fold.events.labeling import BinarizeSign, FixedForwardHorizon
from fold.events.weights import NoWeighting
from fold.loop import train, train_evaluate
from fold.loop.backend.joblib import JoblibBackend
from fold.loop.backend.ray import RayBackend
from fold.loop.backtesting import backtest
from fold.loop.encase import train_backtest
from fold.loop.inference import infer
from fold.models import WrapSKLearnClassifier
from fold.splitters import ExpandingWindowSplitter, SlidingWindowSplitter
from fold.transformations import AddWindowFeatures, RemoveLowVarianceFeatures
from fold.transformations.dev import Identity
from fold.transformations.function import ApplyFunction
from fold.transformations.lags import AddLagsX, AddLagsY
from fold.transformations.scaling import MinMaxScaler
from fold.utils.dataset import get_preprocessed_dataset


@pytest.mark.parametrize(
    "backend",
    [
        "no",
        "thread",
        RayBackend(limit_threads=2),
        JoblibBackend(limit_threads=2, prefer_threads=True),
    ],
)
def test_on_weather_data_backends(backend: str) -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )
    events = FixedForwardHorizon(
        2, BinarizeSign(), weighting_strategy=NoWeighting()
    ).label_events(EveryNth(3).get_event_start_times(y), y)
    splitter = SlidingWindowSplitter(train_window=200, step=200)
    pipeline = PipelineCard(
        preprocessing=[
            Concat(
                [
                    Concat(
                        [
                            Sequence(
                                AddLagsX(
                                    columns_and_lags=[("pressure", list(range(1, 3)))]
                                )
                            ),
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
            Identity(),
            MinMaxScaler(),
        ],
        pipeline=[
            RemoveLowVarianceFeatures(),
            MinMaxScaler(),
            RandomForestRegressor(random_state=42),
        ],
    )

    _, pred, trained_pipeline = train_evaluate(
        pipeline, X, y, splitter, backend=backend, events=events
    )
    assert len(pred) == 799

    inference_output = infer(trained_pipeline, X)
    assert len(inference_output) == len(X)
    assert (
        inference_output.isna().sum() == 0
    )  # inference should be emitting events for all rows


def test_inference() -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )
    splitter = SlidingWindowSplitter(train_window=200, step=200)
    pipeline = PipelineCard(
        preprocessing=[
            Concat(
                [
                    Concat(
                        [
                            Sequence(
                                AddLagsX(
                                    columns_and_lags=[("pressure", list(range(1, 3)))]
                                )
                            ),
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
            Identity(),
            MinMaxScaler(),
        ],
        pipeline=Identity(),
    )

    pred, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    inference_output = infer(trained_pipeline, X)
    assert inference_output.loc[pred.index].equals(pred)


def test_train_evaluate() -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    pipeline = PipelineCard(
        preprocessing=[
            AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))]),
            AddLagsY(list(range(1, 10))),
        ],
        pipeline=RandomForestRegressor(),
    )

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

    pipeline = PipelineCard(
        preprocessing=[
            AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))]),
            AddLagsY(list(range(1, 3))),
        ],
        pipeline=[RandomForestClassifier()],
        event_labeler=FixedForwardHorizon(
            1, labeling_strategy=BinarizeSign(), weighting_strategy=None
        ),
    )

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
    pipeline = PipelineCard(
        preprocessing=[
            AddLagsY(list(range(1, 3))),
            AddLagsX(columns_and_lags=[("pressure", list(range(1, 10)))]),
        ],
        pipeline=[
            OptimizeGridSearch(
                pipeline=SelectBest(
                    [
                        WrapSKLearnClassifier.from_model(LogisticRegression()),
                        WrapSKLearnClassifier.from_model(RandomForestClassifier()),
                    ]
                ),
                krisi_metric_key="f_one_score_macro",
            )
        ],
        event_labeler=FixedForwardHorizon(
            time_horizon=5,
            labeling_strategy=BinarizeSign(),
            weighting_strategy=None,
        ),
        event_filter=EveryNth(2),
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
