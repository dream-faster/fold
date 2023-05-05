from fold.events import BinarizeFixedForwardHorizon, CreateEvents, NoFilter
from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Identity
from fold.utils.tests import generate_sine_wave_data


def test_event() -> None:
    X, y = generate_sine_wave_data()
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = CreateEvents(Identity(), BinarizeFixedForwardHorizon(10), NoFilter())
    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()
