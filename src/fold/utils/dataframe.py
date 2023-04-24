from typing import Union

import pandas as pd


def to_series(dataframe_or_series: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    if isinstance(dataframe_or_series, pd.DataFrame):
        assert len(dataframe_or_series.columns) > 0
        return dataframe_or_series.squeeze()
    elif isinstance(dataframe_or_series, pd.Series):
        return dataframe_or_series
    else:
        raise ValueError("Not a pd.Series or pd.DataFrame")
