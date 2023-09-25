from typing import Callable, Optional

import pandas as pd


def create_forward_rolling(
    transformation_func: Optional[Callable],
    agg_func: Callable,
    series: pd.Series,
    period: int,
    shift_by: Optional[int],
) -> pd.Series:
    assert period > 0
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=period)
    shift_by = shift_by if shift_by is not None else -1
    assert shift_by < 0
    transformation_func = transformation_func if transformation_func else lambda x: x
    return agg_func(transformation_func(series).rolling(window=indexer)).shift(shift_by)
