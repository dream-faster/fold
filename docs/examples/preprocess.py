from fold_models import WrapXGB
from xgboost import XGBRegressor

from fold.composites.target import TransformTarget
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.difference import Difference
from fold.transformations.lags import AddLagsX
from fold.transformations.window import AddWindowFeatures
from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la",
    target_col="temperature",
)

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline = TransformTarget(
    [
        AddWindowFeatures([("temperature", 14, "mean")]),
        Difference(),
        AddLagsX(columns_and_lags=[("temperature", list(range(1, 5)))]),
        WrapXGB.from_model(XGBRegressor()),
    ],
    Difference(),
)
