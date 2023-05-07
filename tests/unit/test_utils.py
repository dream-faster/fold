import numpy as np
import pandas as pd
import pytest

from fold.composites.common import traverse, traverse_apply
from fold.composites.target import TransformTarget
from fold.transformations.dev import Lookahead, Test
from fold.transformations.difference import Difference
from fold.transformations.math import TakeLog
from fold.utils.dataframe import to_series
from fold.utils.trim import (
    get_first_valid_index,
    trim_initial_nans,
    trim_initial_nans_single,
)


def test_get_first_valid_index():
    assert get_first_valid_index(pd.Series([1, 2, 3])) == 0
    assert get_first_valid_index(pd.Series([np.nan, 2, 3])) == 1
    assert get_first_valid_index(pd.Series([np.nan, np.nan, 3])) == 2
    assert get_first_valid_index(pd.Series([np.nan, np.nan, np.nan])) is None
    assert get_first_valid_index(pd.Series([np.nan, np.nan, np.nan, 4])) == 3
    assert get_first_valid_index(pd.Series([np.nan, np.nan, np.nan, 4, 5])) == 3


def test_trim_initial_nans():
    X = pd.DataFrame(
        {
            "a": [np.nan, np.nan, 3, 4, 5],
            "b": [np.nan, np.nan, np.nan, 4, 5],
        }
    )
    y = pd.Series([1, 2, 3, 4, 5])
    trimmed_X, trimmed_y, _ = trim_initial_nans(X, y, y)
    assert trimmed_X.equals(X.iloc[3:])
    assert trimmed_y.equals(y.iloc[3:])
    assert trim_initial_nans_single(X).equals(X.iloc[3:])

    X = pd.DataFrame(
        {
            "a": [0.0, 0.0, 3, 4, 5],
            "b": [np.nan, np.nan, 0.1, 4, 5],
        }
    )
    y = pd.Series([1, 2, 3, 4])
    trimmed_X, trimmed_y, _ = trim_initial_nans(X, y, y)
    assert trimmed_X.equals(X.iloc[2:])
    assert trimmed_y.equals(y.iloc[2:])
    assert trim_initial_nans_single(X).equals(X.iloc[2:])

    X = pd.DataFrame(
        {
            "a": [0.0, 0.0, 3, 4, 5],
            "b": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    y = pd.Series([1, 2, 3, 4])
    trimmed_X, trimmed_y, _ = trim_initial_nans(X, y, y)
    assert len(trimmed_X) == 0
    assert len(trimmed_y) == 0


def test_to_series_df_more_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(ValueError):
        to_series(df)


def test_to_series_df_single_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    series = to_series(df)
    assert series.name == "a"
    assert series.tolist() == [1, 2, 3]


def test_to_series_df_single_column_single_value():
    df = pd.DataFrame({"a": [1]})
    series = to_series(df)
    assert series.name == "a"
    assert series.tolist() == [1]


def test_traverse_apply():
    pipeline = TransformTarget(
        [
            Test(fit_func=lambda x: x, transform_func=lambda x: x),
            Lookahead(),
        ],
        y_pipeline=[TakeLog(), Difference()],
    )

    def set_value_on_lookahead(x):
        if isinstance(x, Lookahead):
            x.value = 1
        return x

    modified_pipeline = traverse_apply(pipeline, set_value_on_lookahead)
    assert modified_pipeline.wrapped_pipeline[0][1].value == 1
    assert not hasattr(pipeline.wrapped_pipeline[0][1], "value")


def test_traverse():
    pipeline = TransformTarget(
        [
            Test(fit_func=lambda x: x, transform_func=lambda x: x),
            Lookahead(),
        ],
        y_pipeline=[TakeLog(), Difference()],
    )

    transformations = list(traverse(pipeline))
    assert len(transformations) == 4
