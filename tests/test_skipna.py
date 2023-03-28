import pandas as pd

from fold.composites.columns import SkipNA
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_all_zeros


def test_skipna() -> None:
    X, y = generate_all_zeros(1000)
    X[70:100] = pd.NA
    y[70:100] = pd.NA

    splitter = ExpandingWindowSplitter(initial_train_window=50, step=400)
    transformations = [
        SkipNA([lambda x: x, lambda x: x]),
    ]

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert pred.squeeze()[20:40].isna().all()
    assert not pred.squeeze()[50:].isna().any()
