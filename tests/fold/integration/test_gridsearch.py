from fold.composites.concat import Concat
from fold.loop.encase import train_backtest
from fold.models.dummy import DummyRegressor
from fold.splitters import ExpandingWindowSplitter, ForwardSingleWindowSplitter
from fold.transformations.features import AddWindowFeatures
from fold.utils.dataset import get_preprocessed_dataset
from fold_extensions.optimize_optuna import OptimizeOptuna


def test_on_weather_data() -> None:
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=100,
    )
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
    pipeline = OptimizeOptuna(
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
        krisi_metric_key="mse",
        is_scorer_loss=True,
        trials=10,
        splitter=ForwardSingleWindowSplitter(0.6),
    )

    _, _, _, _ = train_backtest(pipeline, X, y, splitter)
