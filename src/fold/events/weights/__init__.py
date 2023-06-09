from typing import Optional, Union

import pandas as pd

from ..utils import calculate_rolling_window_size


def calculate_sample_weights(
    returns: pd.Series, window_size: Optional[Union[float, int]]
):
    abs_returns = returns.abs()
    rolling_or_expanding = (
        abs_returns.expanding()
        if window_size is None
        else abs_returns.rolling(
            calculate_rolling_window_size(window_size, abs_returns)
        )
    )
    maximum = rolling_or_expanding.max()
    maximum = maximum.fillna(abs_returns.expanding().max())
    sample_weights = returns.abs() / maximum
    return sample_weights.fillna(0.0)
