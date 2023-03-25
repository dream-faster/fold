from sklearn.ensemble import HistGradientBoostingRegressor

from fold import train_evaluate
from fold.loop import train
from fold.loop.backtesting import backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.lags import AddLagsX, AddLagsY
from fold.utils.dataset import get_preprocessed_dataset


def test_on_weather_data() -> None:
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

    transformations_over_time = train(pipeline, X, y, splitter)
    backtest(transformations_over_time, X, y, splitter)


def test_train_eval_with_krisi() -> None:
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

    # Without splitter
    scorecard, pred, trained_transformations_over_time = train_evaluate(pipeline, X, y)

    # With splitter
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    scorecard, pred, trained_transformations_over_time = train_evaluate(
        pipeline, X, y, splitter
    )
