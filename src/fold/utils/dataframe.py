from functools import reduce
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from .enums import ParsableEnum
from .list import filter_none, flatten, has_intersection, keep_only_duplicates


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


def to_dataframe(dataframe_or_series: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    if isinstance(dataframe_or_series, pd.DataFrame):
        return dataframe_or_series
    elif isinstance(dataframe_or_series, pd.Series):
        return dataframe_or_series.to_frame()
    else:
        raise ValueError("Not a pd.Series or pd.DataFrame")


def __concat_on_axis(axis: str) -> Callable:
    def concat_on(
        dfs: List[Optional[Union[pd.DataFrame, pd.Series]]], copy: bool
    ) -> pd.DataFrame:
        filtered = filter_none(dfs)
        if len(filtered) == 0:
            return None  # type: ignore
        elif len(filtered) == 1:
            return filtered[0]
        else:
            return pd.concat(filtered, axis=axis, copy=copy)

    return concat_on


concat_on_columns = __concat_on_axis("columns")
concat_on_index = __concat_on_axis("index")


def concat_on_index_override_duplicate_rows(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if len(dfs) == 0:
        return pd.DataFrame()
    elif len(dfs) == 1:
        return dfs[0]
    else:
        return pd.concat(dfs, axis="index", copy=False).groupby(level=0).last()


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


class ResolutionStrategy(ParsableEnum):
    """
    Parameters
    ----------
    first : str
        Only keep the first (leftmost) duplicate column(s).
    last : str
        Only keep the last (rightmost) duplicate column(s).
    both : str
        Keep both duplicate columns.
    """

    first = "first"
    last = "last"
    both = "both"


def concat_on_columns_with_duplicates(
    dfs: List[Union[pd.DataFrame, pd.Series]],
    strategy: ResolutionStrategy,
    raise_indices_dont_match: bool = False,
) -> pd.DataFrame:
    if len(dfs) == 0:
        return pd.DataFrame()
    if all([result.empty for result in dfs]):
        return pd.DataFrame(index=dfs[0].index)
    dfs = [to_dataframe(df) for df in dfs]
    max_len = max([len(df) for df in dfs])
    if not all([len(df) == max_len for df in dfs]):
        index_to_apply = reduce(
            lambda i1, i2: i1.union(i2), [df.index for df in dfs]
        ).sort_values()
        dfs = [df.reindex(index_to_apply) for df in dfs]

    columns = flatten([result.columns.to_list() for result in dfs])
    duplicates = keep_only_duplicates(columns)
    if len(duplicates) == 0:
        return pd.concat(dfs, axis="columns", copy=False)

    if len(duplicates) > 0 or strategy is not ResolutionStrategy.both:

        def concat(
            dfs: List[pd.DataFrame],
        ) -> pd.DataFrame:
            if not raise_indices_dont_match and not all(
                [df.index.equals(dfs[0].index) for df in dfs]
            ):
                return reduce(lambda accum, item: accum.combine_first(item), dfs)
            else:
                return pd.concat(dfs, copy=False, axis="columns")

        duplicate_columns = [
            result[
                keep_only_duplicates(flatten([duplicates, result.columns.to_list()]))
            ]
            for result in dfs
            if has_intersection(result.columns.to_list(), duplicates)
        ]
        results = [result.drop(columns=duplicates, errors="ignore") for result in dfs]
        results = [
            df for df in results if not df.empty
        ]  # if all of them are empty, create an empty list

        if strategy is ResolutionStrategy.first:
            return concat(results + [duplicate_columns[0]])
        elif strategy is ResolutionStrategy.last:
            return concat(
                results + [duplicate_columns[-1]],
            )
        elif strategy is ResolutionStrategy.both:
            return concat(results + duplicate_columns)
        else:
            raise ValueError(f"ResolutionStrategy is not valid: {strategy}")
    else:
        return pd.concat(dfs, copy=False, axis="columns")


def fill_na_inf(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def apply_function_batched(
    df: pd.DataFrame, func: Callable, batch_columns: Optional[int]
) -> pd.DataFrame:
    if batch_columns is not None:
        batches = [
            func(df.iloc[:, i : i + batch_columns])
            for i in range(0, len(df.columns), batch_columns)
        ]
        return pd.concat(batches, axis="columns", copy=False)
    else:
        return func(df)
