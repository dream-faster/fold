import numpy as np
import pandas as pd

from fold.utils.forward import create_forward_rolling


def test_create_forward_rolling():
    series = pd.Series([1, 2, 3, 4, 5])
    period = 1
    shift_by = None
    expected_output = pd.Series([2.0, 3.0, 4.0, 5.0, np.nan])
    output = create_forward_rolling(
        None,
        pd.core.window.rolling.Rolling.mean,
        series,
        period,
        shift_by,
    )
    assert output.equals(expected_output)

    series = pd.Series([1, 2, 3, 4, 5])
    period = 1
    extra_shift_by = -1
    expected_output = pd.Series([3.0, 4.0, 5.0, np.nan, np.nan])
    output = create_forward_rolling(
        None,
        pd.core.window.rolling.Rolling.mean,
        series,
        period,
        extra_shift_by,
    )
    assert output.equals(expected_output)

    series = pd.Series([1, 2, 3, 4, 5])
    period = 2
    extra_shift_by = None
    expected_output = pd.Series([2.5, 3.5, 4.5, np.nan, np.nan])
    output = create_forward_rolling(
        None,
        pd.core.window.rolling.Rolling.mean,
        series,
        period,
        extra_shift_by,
    )
    assert output.equals(expected_output)
