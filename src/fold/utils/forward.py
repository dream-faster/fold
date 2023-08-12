from typing import Callable, Optional

import pandas as pd


def create_forward_rolling(
    agg_func: Callable, series: pd.Series, period: int, shift_by: Optional[int]
) -> pd.Series:
    assert period > 0
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=period)
    shift_by = shift_by if shift_by is not None else -int(max(1, period / 2))
    return agg_func(series.rolling(window=indexer)).shift(shift_by)
