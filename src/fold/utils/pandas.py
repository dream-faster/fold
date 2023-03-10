from typing import Tuple, Union

import numpy as np
import pandas as pd


def trim_initial_nans(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    # Optimize for speed, if the first value is not NaN, we can save all the subsequent computation
    if not X.iloc[0].isna().any():
        return X, y
    first_valid_index = get_first_valid_index(X)
    return X.iloc[first_valid_index:], y.iloc[first_valid_index:]


def trim_initial_nans_single(X: pd.DataFrame) -> pd.DataFrame:
    # Optimize for speed, if the first value is not NaN, we can save all the subsequent computation
    if not X.iloc[0].isna().any():
        return X
    first_valid_index = get_first_valid_index(X)
    return X.iloc[first_valid_index:]


def get_first_valid_index(series: Union[pd.Series, pd.DataFrame]) -> int:
    double_nested_results = np.where(np.logical_not(pd.isna(series)))
    if len(double_nested_results) == 0:
        return 0
    nested_result = double_nested_results[0]
    if len(nested_result) == 0:
        return 0
    return nested_result[0]
