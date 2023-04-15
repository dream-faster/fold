"""
Using Tabular Models
===========================
"""

from sklearn.ensemble import RandomForestRegressor

from fold.composites import Concat
from fold.loop import train_evaluate
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import AddLagsX, AddWindowFeatures
from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la", target_col="temperature", shorten=1000
)

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline = [
    Concat(
        [
            AddWindowFeatures([("temperature", 14, "mean")]),
            AddLagsX(columns_and_lags=[("temperature", list(range(1, 5)))]),
        ]
    ),
    RandomForestRegressor(),
]

scorecard, prediction, trained_pipelines = train_evaluate(pipeline, X, y, splitter)
