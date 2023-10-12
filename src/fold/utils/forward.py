from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.core.indexers.objects import BaseIndexer


class FixedForwardWindowIndexerNoTruncation(BaseIndexer):
    """
    Creates window boundaries for fixed-length windows that include the current row.

    Examples
    --------
    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    >>> df.rolling(window=indexer, min_periods=1).sum()
         B
    0  1.0
    1  3.0
    2  2.0
    3  4.0
    4  4.0
    """

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if center:
            raise ValueError("Forward-looking windows can't have center=True")
        if closed is not None:
            raise ValueError(
                "Forward-looking windows don't support setting the closed argument"
            )
        if step is None:
            step = 1

        start = np.arange(0, num_values, step, dtype="int64")
        end = start + self.window_size

        return start, end


def create_forward_rolling(
    transformation_func: Optional[Callable],
    agg_func: Callable,
    series: pd.Series,
    period: int,
    shift_by: Optional[int],
    truncate_end: bool,
) -> pd.Series:
    assert period > 0
    indexer = (
        pd.api.indexers.FixedForwardWindowIndexer(window_size=period)
        if truncate_end
        else FixedForwardWindowIndexerNoTruncation(window_size=period)
    )
    shift_by = shift_by if shift_by is not None else -1
    assert shift_by < 0
    transformation_func = transformation_func if transformation_func else lambda x: x
    return agg_func(
        transformation_func(series).rolling(window=indexer, min_periods=1)
    ).shift(shift_by)
