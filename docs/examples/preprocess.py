"""
Preprocessing
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'
from fold import train
from fold.composites import Concat
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import AddLagsX, AddLagsY, AddWindowFeatures, Difference
from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la",
    target_col="temperature",
)

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline = [
    Difference(),
    Concat(
        [
            AddWindowFeatures([("temperature", 14, "mean")]),
            AddLagsX(columns_and_lags=[("temperature", list(range(1, 5)))]),
            AddLagsY([1, 2]),
        ]
    ),
]


trained_pipelines = train(pipeline, X, y, splitter)
