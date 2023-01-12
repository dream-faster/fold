import pytest

from drift.loop import walk_forward_inference, walk_forward_train
from drift.models import Baseline, BaselineStrategy
from drift.splitters import ExpandingWindowSplitter
from drift.transformations import NoTransformation
from tests.utils import generate_sine_wave_data


def test_baseline_naive_model() -> None:

    y = generate_sine_wave_data()
    X = y.shift(1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [NoTransformation(), Baseline(strategy=BaselineStrategy.naive)]

    transformations_over_time = walk_forward_train(transformations, X, y, splitter)
    _, pred = walk_forward_inference(transformations_over_time, X, splitter)
    assert (y[pred.index].shift(1) == pred).sum() == len(pred) - 1
