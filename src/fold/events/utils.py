from typing import Union

import pandas as pd


def calculate_rolling_window_size(
    window_size: Union[int, float], series: pd.Series
) -> int:
    return window_size if window_size > 1 else int(len(series) * window_size)  # type: ignore
