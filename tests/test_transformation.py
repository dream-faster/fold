import pytest

from drift.loop import fit_transformations, walk_forward_inference, walk_forward_train
from drift.models import BaselineModel, BaselineStrategy
from drift.transformations.no import NoTransformation
from drift.utils.splitters import ExpandingWindowSplitter
from tests.utils import generate_sine_wave_data


def test_baseline_naive_model() -> None:

    y = generate_sine_wave_data()
    X = y.shift(1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    model = BaselineModel(strategy=BaselineStrategy.naive)
    transformation = NoTransformation()

    transformations_over_time = fit_transformations(X, y, splitter, transformation)
    model_over_time = walk_forward_train(model, X, y, splitter, transformations_over_time)

    _, pred = walk_forward_inference(model_over_time, transformations_over_time, X, y, splitter)
    assert (y[pred.index].shift(1) == pred).sum() == len(pred) - 1
