from sklearn.metrics import mean_squared_error

from fold.composites.concat import Concat
from fold.composites.optimize import OptimizeGridSearch
from fold.loop.encase import train_backtest
from fold.models.dummy import DummyRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.window import AddWindowFeatures
from fold.utils.dataset import get_preprocessed_dataset


def test_on_weather_data() -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=100,
    )
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    pipeline = OptimizeGridSearch(
        [
            Concat(
                [
                    AddWindowFeatures(("all", 14, "mean")),
                    AddWindowFeatures(("all", 14, "std")),
                ]
            ),
            DummyRegressor(
                predicted_value=1.0, params_to_try=dict(predicted_value=[1.0, 2.0])
            ),
        ],
        scorer=mean_squared_error,
    )

    _, _ = train_backtest(pipeline, X, y, splitter)
