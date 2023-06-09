from fold.events import CreateEvents, FixedForwardHorizon, UsePredefinedEvents
from fold.events.filters.everynth import EveryNth
from fold.events.labeling import BinarizeSign
from fold.loop import backtest, train
from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Identity
from fold.utils.tests import generate_sine_wave_data


def test_create_event() -> None:
    X, y = generate_sine_wave_data(length=1100)
    splitter = ExpandingWindowSplitter(initial_train_window=100, step=200)
    pipeline = CreateEvents(
        Identity(),
        FixedForwardHorizon(
            1, labeling_strategy=BinarizeSign(), weighing_strategy=None
        ),
        EveryNth(5),
    )
    trained_pipeline, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    pred = backtest(trained_pipeline, X, y, splitter)
    assert len(pred) == 1000
    assert len(pred.dropna()) == 200

    pipeline = CreateEvents(
        Identity(),
        FixedForwardHorizon(
            10, labeling_strategy=BinarizeSign(), weighing_strategy=None
        ),
        EveryNth(5),
    )
    trained_pipeline, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    pred = backtest(trained_pipeline, X, y, splitter)
    assert len(pred) == 1000
    assert len(pred.dropna()) == 190


def test_predefined_events() -> None:
    X, y = generate_sine_wave_data(length=1100)
    splitter = ExpandingWindowSplitter(initial_train_window=100, step=200)

    event_filter = EveryNth(5)
    labeler = FixedForwardHorizon(
        2, labeling_strategy=BinarizeSign(), weighing_strategy=None
    )

    original_start_times = event_filter.get_event_start_times(y)
    events = labeler.label_events(original_start_times, y).reindex(y.index)

    pipeline = UsePredefinedEvents(Identity())
    pred, trained_pipeline, artifacts = train_backtest(
        pipeline, X, y, splitter, events=events, return_artifacts=True
    )
    assert len(pred) == 1000
    assert len(pred.dropna()) == 200
