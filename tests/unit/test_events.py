import pytest
from sklearn.ensemble import RandomForestRegressor

from fold.base.classes import PipelineCard
from fold.events import FixedForwardHorizon, UsePredefinedEvents
from fold.events.filters.everynth import EveryNth
from fold.events.filters.no import NoFilter
from fold.events.labeling import BinarizeSign
from fold.events.labeling.strategies import NoLabel
from fold.events.weights import NoWeighting
from fold.loop import backtest, train
from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Identity
from fold.transformations.lags import AddLagsY
from fold.utils.tests import generate_sine_wave_data


def test_create_event() -> None:
    X, y = generate_sine_wave_data(length=1100)
    splitter = ExpandingWindowSplitter(initial_train_window=100, step=200)
    pipeline = PipelineCard(
        preprocessing=None,
        pipeline=Identity(),
        event_labeler=FixedForwardHorizon(
            1, labeling_strategy=BinarizeSign(), weighting_strategy=None
        ),
        event_filter=EveryNth(5),
    )
    trained_pipeline, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    pred = backtest(trained_pipeline, X, y, splitter)
    assert len(pred) == 1000
    assert len(pred.dropna()) == 200

    pipeline = PipelineCard(
        preprocessing=None,
        pipeline=Identity(),
        event_labeler=FixedForwardHorizon(
            10, labeling_strategy=BinarizeSign(), weighting_strategy=None
        ),
        event_filter=EveryNth(5),
    )
    trained_pipeline, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    pred = backtest(trained_pipeline, X, y, splitter)
    assert len(pred) == 1000
    assert len(pred.dropna()) == 190


@pytest.mark.parametrize("agg_func", ["mean", "std", "max", "min"])
def test_predefined_events(agg_func: str) -> None:
    X, y = generate_sine_wave_data(length=1100)
    splitter = ExpandingWindowSplitter(initial_train_window=100, step=200)

    labeler = FixedForwardHorizon(
        2,
        labeling_strategy=NoLabel(),
        weighting_strategy=NoWeighting(),
        aggregate_function=agg_func,
    )

    original_start_times = y.index
    events = labeler.label_events(original_start_times, y).reindex(y.index)

    model = [AddLagsY([1]), RandomForestRegressor(random_state=0)]
    usepredefined_pipeline = UsePredefinedEvents(model)
    usepredefined_pred, _, usepredefined_artifact = train_backtest(
        usepredefined_pipeline, X, y, splitter, events=events, return_artifacts=True
    )
    assert len(usepredefined_pred) == 1000
    assert len(usepredefined_pred.dropna()) == 998
    assert len(usepredefined_artifact) == len(X)

    pipeline_card = PipelineCard(
        preprocessing=None,
        pipeline=model,
        event_labeler=labeler,
        event_filter=NoFilter(),
    )
    pred, _, artifact = train_backtest(
        pipeline_card, X, y, splitter, return_artifacts=True
    )

    assert len(pred) == 1000
    assert len(pred.dropna()) == 998
    assert pred.equals(usepredefined_pred)
    assert len(artifact) == len(X)
