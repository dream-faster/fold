from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from fold.utils.list import filter_none


def to_series(dataframe_or_series: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    if isinstance(dataframe_or_series, pd.DataFrame):
        if len(dataframe_or_series.columns) != 1:
            raise ValueError("DataFrame must have exactly one column")
        if len(dataframe_or_series) == 1:
            return dataframe_or_series[dataframe_or_series.columns[0]]
        else:
            return dataframe_or_series.squeeze()
    elif isinstance(dataframe_or_series, pd.Series):
        return dataframe_or_series
    else:
        raise ValueError("Not a pd.Series or pd.DataFrame")


def __concat_on_axis(axis: str) -> Callable:
    def concat_on(dfs: List[Optional[Union[pd.DataFrame, pd.Series]]]) -> pd.DataFrame:
        filtered = filter_none(dfs)
        if len(filtered) == 0:
            return None  # type: ignore
        elif len(filtered) == 1:
            return filtered[0]
        else:
            return pd.concat(filtered, axis=axis)

    return concat_on


concat_on_columns = __concat_on_axis("columns")
concat_on_index = __concat_on_axis("index")


def take_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    If you use np.log() on a DataFrame with a single column, it'll turn it into a pd.Series.
    This function retains the original instance type (pd.DataFrame).
    """
    result = np.log(df)
    if isinstance(df, pd.Series):
        return result.to_frame()
    else:
        return result
