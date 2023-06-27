from typing import Callable

import pandas as pd


def create_forward_rolling(
    agg_func: Callable, series: pd.Series, period: int
) -> pd.Series:
    assert period > 0
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=period)
    return agg_func(series.rolling(window=indexer)).shift(-1)
