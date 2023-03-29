from sklearn.ensemble import RandomForestRegressor

from fold.loop import train_evaluate
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import AddDateTimeFeatures, AddHolidayFeatures
from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la", target_col="temperature", shorten=1000
)

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline = [
    AddDateTimeFeatures(["hour", "day_of_week"]),
    AddHolidayFeatures(["US"]),
    RandomForestRegressor(),
]

scorecard, prediction, trained_pipelines = train_evaluate(pipeline, X, y, splitter)
