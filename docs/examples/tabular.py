from fold_models import WrapXGB
from xgboost import XGBRegressor

from fold.loop import train_evaluate
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.difference import Difference
from fold.transformations.lags import AddLagsX
from fold.transformations.window import AddWindowFeatures
from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la", target_col="temperature", shorten=1000
)

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline = [
    Difference(),
    AddWindowFeatures([("temperature", 14, "mean")]),
    AddLagsX(columns_and_lags=[("temperature", list(range(1, 5)))]),
    WrapXGB.from_model(XGBRegressor()),
]

scorecard, prediction, trained_pipelines = train_evaluate(pipeline, X, y, splitter)
