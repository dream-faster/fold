from typing import List, Optional, Union

import pandas as pd

from .list import filter_none


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


def concat_on_index(dfs: List[Optional[pd.DataFrame]]) -> pd.DataFrame:
    return pd.concat(filter_none(dfs), axis="index")


def concat_on_columns(dfs: List[Optional[pd.DataFrame]]) -> pd.DataFrame:
    return pd.concat(filter_none(dfs), axis="columns")
