from collections.abc import Callable

import pandas as pd


def create_forward_rolling(
    transformation_func: Callable | None,
    agg_func: Callable,
    series: pd.Series,
    period: int,
    extra_shift_by: int | None,
) -> pd.Series:
    assert period > 0
    extra_shift_by = abs(extra_shift_by) if extra_shift_by is not None else 0
    transformation_func = transformation_func if transformation_func else lambda x: x
    return agg_func(
        transformation_func(series).rolling(window=period, min_periods=1)
    ).shift(-period - extra_shift_by)
