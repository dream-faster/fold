from collections.abc import Callable
from functools import reduce

import numpy as np
import pandas as pd
from finml_utils.dataframes import concat_on_columns
from finml_utils.enums import ParsableEnum
from tqdm import tqdm

from .list import flatten, has_intersection, keep_only_duplicates


def to_series(dataframe_or_series: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(dataframe_or_series, pd.DataFrame):
        if len(dataframe_or_series.columns) != 1:
            raise ValueError("DataFrame must have exactly one column")
        if len(dataframe_or_series) == 1:
            return dataframe_or_series[dataframe_or_series.columns[0]]
        return dataframe_or_series.squeeze()
    if isinstance(dataframe_or_series, pd.Series):
        return dataframe_or_series
    raise ValueError("Not a pd.Series or pd.DataFrame")


def to_dataframe(dataframe_or_series: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(dataframe_or_series, pd.DataFrame):
        return dataframe_or_series
    if isinstance(dataframe_or_series, pd.Series):
        return dataframe_or_series.to_frame()
    raise ValueError("Not a pd.Series or pd.DataFrame")


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
    dfs: list[pd.DataFrame | pd.Series],
    strategy: ResolutionStrategy,
    raise_indices_dont_match: bool = False,
) -> pd.DataFrame:
    if len(dfs) == 0:
        return pd.DataFrame()
    if all(result.empty for result in dfs):
        return pd.DataFrame(index=dfs[0].index)
    dfs = [to_dataframe(df) for df in dfs]
    max_len = max([len(df) for df in dfs])
    if not all(len(df) == max_len for df in dfs):
        index_to_apply = reduce(
            lambda i1, i2: i1.union(i2), [df.index for df in dfs]
        ).sort_values()
        dfs = [df.reindex(index_to_apply) for df in dfs]

    columns = flatten([result.columns.to_list() for result in dfs])
    duplicates = keep_only_duplicates(columns)
    if (
        len(duplicates) == 0
        or strategy is ResolutionStrategy.both
        or len(duplicates) == 0
    ):
        return concat_on_columns(dfs)

    def concat(
        dfs: list[pd.DataFrame],
    ) -> pd.DataFrame:
        if not raise_indices_dont_match and not all(
            df.index.equals(dfs[0].index) for df in dfs
        ):
            return reduce(lambda accum, item: accum.combine_first(item), dfs)
        return concat_on_columns(dfs)

    duplicate_columns = [
        result[keep_only_duplicates(flatten([duplicates, result.columns.to_list()]))]
        for result in dfs
        if has_intersection(result.columns.to_list(), duplicates)
    ]
    results = [result.drop(columns=duplicates, errors="ignore") for result in dfs]
    results = [
        df for df in results if not df.empty
    ]  # if all of them are empty, create an empty list

    if strategy is ResolutionStrategy.first:
        return concat([*results, duplicate_columns[0]])
    if strategy is ResolutionStrategy.last:
        return concat(
            [*results, duplicate_columns[-1]],
        )
    if strategy is ResolutionStrategy.both:
        return concat(results + duplicate_columns)
    raise ValueError(f"ResolutionStrategy is not valid: {strategy}")


def fill_na_inf(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def apply_function_batched(
    df: pd.DataFrame,
    func: Callable,
    batch_columns: int | None,
    display_progress: bool = False,
) -> pd.DataFrame:
    if batch_columns is not None:
        batches = [
            func(df.iloc[:, i : i + batch_columns])
            for i in tqdm(
                range(0, len(df.columns), batch_columns), disable=not display_progress
            )
        ]
        return concat_on_columns(batches)
    return func(df)
