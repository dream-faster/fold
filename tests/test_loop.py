import pytest

from drift.loop import infer, train
from drift.models import Baseline, BaselineStrategy
from drift.splitters import ExpandingWindowSplitter
from drift.transformations import NoTransformation
from tests.utils import generate_sine_wave_data


def test_loop() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [Baseline(strategy=BaselineStrategy.naive)]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert (X[pred.index] == pred).sum() == len(pred) - 1
