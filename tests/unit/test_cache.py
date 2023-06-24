from fold.composites import Cache
from fold.events import CreateEvents
from fold.events.filters.everynth import EveryNth
from fold.events.labeling import FixedForwardHorizon, NoLabel
from fold.events.weights import NoWeighing
from fold.loop import train_backtest
from fold.models.random import RandomClassifier
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data


def test_cache() -> None:
    X, y = generate_monotonous_data(1000, freq="1min")

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
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
            path="tests/unit/cache/",
        )
    ]
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
