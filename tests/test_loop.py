import pytest

from src.drift.loop import infer, train
from src.drift.models import Baseline, BaselineStrategy
from src.drift.splitters import ExpandingWindowSplitter
from tests.utils import generate_sine_wave_data


def test_loop() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [Baseline(strategy=BaselineStrategy.naive)]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert (X.squeeze()[pred.index] == pred).all()
