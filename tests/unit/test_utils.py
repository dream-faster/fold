import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression

from fold.base import Composite, traverse, traverse_apply
from fold.base.classes import Extras
from fold.composites.target import TransformTarget
from fold.models.base import postpostprocess_output
from fold.transformations.dev import Lookahead, Test
from fold.transformations.difference import Difference
from fold.transformations.math import TakeLog
from fold.transformations.sklearn import WrapSKLearnFeatureSelector
from fold.utils.dataframe import to_series
from fold.utils.tests import generate_all_zeros
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
    trimmed_X, trimmed_y, _ = trim_initial_nans(X, y, Extras(sample_weights=y))
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
    trimmed_X, trimmed_y, _ = trim_initial_nans(X, y, Extras(sample_weights=y))
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
    trimmed_X, trimmed_y, _ = trim_initial_nans(X, y, Extras(sample_weights=y))
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

    def set_value_on_lookahead(x, clone_children):
        if isinstance(x, Lookahead):
            x = Lookahead()
            x.value = 1
            return x
        elif isinstance(x, Composite):
            return x.clone(clone_children)
        else:
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
    assert len(transformations) == 5


def test_unique_id_per_instance() -> None:
    transformations = [
        Difference(),
        Difference(),
    ]
    assert transformations[0].id != transformations[1].id

    transformations = [
        WrapSKLearnFeatureSelector.from_model(VarianceThreshold()),
        WrapSKLearnFeatureSelector.from_model(
            SelectKBest(score_func=f_regression, k=1),
        ),
    ]
    assert transformations[0].id != transformations[1].id


def test_sort_probabilities_columns() -> None:
    X, y = generate_all_zeros()
    X["predicitions_a"] = y
    X["probabilities_a_1"] = y
    X["probabilities_a_0"] = y
    assert X.columns[2] == "probabilities_a_1"
    assert X.columns[3] == "probabilities_a_0"
    X.columns = list(map(str, X.columns))

    result = postpostprocess_output(X, name="a")
    assert result.columns[2] == "probabilities_a_0"
    assert result.columns[3] == "probabilities_a_1"
