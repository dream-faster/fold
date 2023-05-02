from fold_models import MovingAverage
from sklearn.metrics import mean_squared_error

from fold.composites.optimize import OptimizeGridSearch
from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.utils.dataset import get_preprocessed_dataset


def test_on_weather_data() -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=100,
    )
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    pipeline = OptimizeGridSearch(
        MovingAverage(window_size=12),
        param_grid={"window_size": range(10, 19)},
        scorer=mean_squared_error,
    )

    _, _ = train_backtest(pipeline, X, y, splitter)
