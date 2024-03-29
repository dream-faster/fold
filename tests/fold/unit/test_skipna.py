import pandas as pd

from fold.composites.columns import SkipNA
from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_zeros_and_ones


def test_skipna() -> None:
    X, y = generate_zeros_and_ones(1000)
    X[70:100] = pd.NA
    y[70:100] = pd.NA

    splitter = ExpandingWindowSplitter(initial_train_window=50, step=400)
    transformations = [
        SkipNA([lambda x: x, lambda x: x]),
    ]

    pred, _, _, _ = train_backtest(transformations, X, y, splitter)
    assert pred.squeeze()[20:40].isna().all()
    assert not pred.squeeze()[50:].isna().any()
