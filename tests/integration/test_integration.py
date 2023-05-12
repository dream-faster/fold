import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from fold.composites import Concat
from fold.loop import train_evaluate
from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import AddWindowFeatures
from fold.transformations.lags import AddLagsX, AddLagsY
from fold.utils.dataset import get_preprocessed_dataset


@pytest.mark.parametrize("backend", ["no", "ray", "thread", "pathos"])
def test_on_weather_data_backends(backend: str) -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    pipeline = [
        Concat(
            [
                AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))]),
                AddLagsY(list(range(1, 10))),
                AddWindowFeatures(("pressure", 14, "mean")),
            ]
        ),
        HistGradientBoostingRegressor(),
    ]

    pred, _ = train_backtest(pipeline, X, y, splitter, backend=backend)
    assert len(pred) == 800


def test_train_evaluate() -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    pipeline = [
        AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))]),
        AddLagsY(list(range(1, 10))),
        HistGradientBoostingRegressor(),
    ]

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    scorecard, pred, trained_trained_pipelines = train_evaluate(
        pipeline, X, y, splitter
    )
