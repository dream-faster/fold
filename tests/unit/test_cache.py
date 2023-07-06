import os
import shutil

import pytest

from fold.composites import Cache
from fold.events import CreateEvents
from fold.events.filters.everynth import EveryNth
from fold.events.labeling import FixedForwardHorizon, NoLabel
from fold.events.weights import NoWeighing
from fold.loop import train_backtest
from fold.models.random import RandomClassifier
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.scaling import StandardScaler
from fold.utils.tests import generate_monotonous_data

folder = "tests/unit/cache/"
os.makedirs("tests/unit/cache/", exist_ok=True)

events_pipeline = [
    Cache(
        CreateEvents(
            RandomClassifier(all_classes=[0, 1]),
            FixedForwardHorizon(
                time_horizon=3,
                labeling_strategy=NoLabel(),
                weighing_strategy=NoWeighing(),
            ),
            EveryNth(2),
        ),
        path=folder,
    )
]

normal_pipeline = [
    Cache(
        RandomClassifier(all_classes=[0, 1]),
        path=folder,
    ),
    StandardScaler(),
]


@pytest.mark.parametrize("pipeline", [events_pipeline, normal_pipeline])
def test_cache(pipeline) -> None:
    X, y = generate_monotonous_data(1000, freq="1min")

    shutil.rmtree(folder)

    splitter = ExpandingWindowSplitter(initial_train_window=200, step=200)
    pred_first, _, artifacts_first = train_backtest(
        pipeline, X, y, splitter, return_artifacts=True
    )
    assert pred_first is not None
    assert artifacts_first is not None
    assert artifacts_first.index.duplicated().sum() == 0

    pred_second, _, artifacts_second = train_backtest(
        pipeline, X, y, splitter, return_artifacts=True
    )
    assert pred_first.equals(pred_second)
    assert artifacts_first.equals(artifacts_second)
