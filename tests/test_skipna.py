import pandas as pd

from drift.loop import backtest, train
from drift.models.dummy import DummyClassifier
from drift.splitters import ExpandingWindowSplitter
from drift.transformations.columns import SkipNA
from drift.utils.tests import generate_all_zeros


def test_skipna() -> None:

    X = generate_all_zeros(1000)
    X[:100] = pd.NA
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=50, step=400)
    transformations = [
        SkipNA([lambda x: x, lambda x: x]),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert pred.squeeze()[:50].isna().all()
    assert pred.squeeze()[50:].isna().any() == False
