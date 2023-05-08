from fold.events import BinarizeFixedForwardHorizon, CreateEvents
from fold.events.filters.everynth import EveryNth
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Identity
from fold.utils.tests import generate_sine_wave_data


def test_event() -> None:
    X, y = generate_sine_wave_data(length=1100)
    splitter = ExpandingWindowSplitter(initial_train_window=100, step=200)
    pipeline = CreateEvents(Identity(), BinarizeFixedForwardHorizon(1), EveryNth(5))
    trained_pipeline, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    pred = backtest(trained_pipeline, X, y, splitter)
    assert len(pred) == 1000
    assert len(pred.dropna()) == 200

    pipeline = CreateEvents(Identity(), BinarizeFixedForwardHorizon(10), EveryNth(5))
    trained_pipeline, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    pred = backtest(trained_pipeline, X, y, splitter)
    assert len(pred) == 1000
    assert len(pred.dropna()) == 190