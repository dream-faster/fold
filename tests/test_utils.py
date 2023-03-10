import numpy as np
import pandas as pd

from fold.utils.pandas import (
    get_first_valid_index,
    trim_initial_nans,
    trim_initial_nans_single,
)


def test_get_first_valid_index():
    assert get_first_valid_index(pd.Series([1, 2, 3])) == 0
    assert get_first_valid_index(pd.Series([np.nan, 2, 3])) == 1
    assert get_first_valid_index(pd.Series([np.nan, np.nan, 3])) == 2
    assert get_first_valid_index(pd.Series([np.nan, np.nan, np.nan])) == 0
    assert get_first_valid_index(pd.Series([np.nan, np.nan, np.nan, 4])) == 3
    assert get_first_valid_index(pd.Series([np.nan, np.nan, np.nan, 4, 5])) == 3


def test_trim_initial_nans():
    X = pd.DataFrame(
        {
            "a": [np.nan, np.nan, 3, 4],
            "b": [np.nan, np.nan, np.nan, 4],
        }
    )
    y = pd.Series([1, 2, 3, 4])
    trimmed_X, trimmed_y = trim_initial_nans(X, y)
    assert trimmed_X.equals(X.iloc[2:])
    assert trimmed_y.equals(y.iloc[2:])
    assert trim_initial_nans_single(X).equals(X.iloc[2:])
