from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression

from fold.base import Composite, traverse, traverse_apply
from fold.composites.concat import Concat
from fold.composites.target import TransformTarget
from fold.composites.utils import _clean_params
from fold.models.base import postpostprocess_output
from fold.transformations.dev import Identity, Test
from fold.transformations.difference import Difference, StationaryMethod
from fold.transformations.math import TakeLog
from fold.transformations.sklearn import WrapSKLearnFeatureSelector
from fold.utils.checks import get_column_names
from fold.utils.dataframe import apply_function_batched, to_series
from fold.utils.tests import generate_zeros_and_ones
from fold.utils.trim import get_first_valid_index


def test_get_first_valid_index():
    assert get_first_valid_index(pd.Series([1, 2, 3])) == 0
    assert get_first_valid_index(pd.Series([np.nan, 2, 3])) == 1
    assert get_first_valid_index(pd.Series([np.nan, np.nan, 3])) == 2
    assert get_first_valid_index(pd.Series([np.nan, np.nan, np.nan])) is None
    assert get_first_valid_index(pd.Series([np.nan, np.nan, np.nan, 4])) == 3
    assert get_first_valid_index(pd.Series([np.nan, np.nan, np.nan, 4, 5])) == 3


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
            Identity(),
        ],
        y_pipeline=[TakeLog(), Difference(method=StationaryMethod.difference)],
    )

    def set_value_on_lookahead(x, clone_children):
        if isinstance(x, Identity):
            x = Identity()
            x.value = 1
            return x
        if isinstance(x, Composite):
            return x.clone(clone_children)
        return x

    modified_pipeline = traverse_apply(pipeline, set_value_on_lookahead)
    assert modified_pipeline.wrapped_pipeline[0][1].value == 1
    assert not hasattr(pipeline.wrapped_pipeline[0][1], "value")


def test_traverse():
    pipeline = TransformTarget(
        [
            Test(fit_func=lambda x: x, transform_func=lambda x: x),
            Identity(),
        ],
        y_pipeline=[TakeLog(), Difference(method=StationaryMethod.difference)],
    )

    transformations = list(traverse(pipeline))
    assert len(transformations) == 5


def test_unique_id_per_instance() -> None:
    transformations = [
        Difference(method=StationaryMethod.difference),
        Difference(method=StationaryMethod.difference),
    ]
    assert transformations[0].id != transformations[1].id

    transformations = [
        WrapSKLearnFeatureSelector.from_model(VarianceThreshold()),
        WrapSKLearnFeatureSelector.from_model(
            SelectKBest(score_func=f_regression, k=1),
        ),
    ]
    assert transformations[0].id != transformations[1].id


def test_cloning() -> None:
    transformations = [
        Difference(method=StationaryMethod.difference),
        Difference(method=StationaryMethod.difference),
    ]
    assert deepcopy(transformations[0]).id == transformations[0].id
    assert deepcopy(transformations[1]).id == transformations[1].id
    assert deepcopy(transformations)[0].id == transformations[0].id
    assert deepcopy(transformations)[1].id == transformations[1].id

    transformations = [
        WrapSKLearnFeatureSelector.from_model(VarianceThreshold()),
        WrapSKLearnFeatureSelector.from_model(
            SelectKBest(score_func=f_regression, k=1),
        ),
    ]
    assert deepcopy(transformations[0]).id == transformations[0].id
    assert deepcopy(transformations[1]).id == transformations[1].id
    assert deepcopy(transformations)[0].id == transformations[0].id
    assert deepcopy(transformations)[1].id == transformations[1].id

    composites = [
        Concat(
            [
                Difference(method=StationaryMethod.difference),
                Difference(method=StationaryMethod.difference),
            ]
        ),
        Concat(
            [
                Difference(method=StationaryMethod.difference),
                Difference(method=StationaryMethod.difference),
            ]
        ),
    ]
    assert deepcopy(composites[0]).id == composites[0].id
    assert deepcopy(composites[1]).id == composites[1].id
    assert composites[0].clone(deepcopy).id == composites[0].id


def test_sort_probabilities_columns() -> None:
    X, y = generate_zeros_and_ones()
    X["predicitions_a"] = y
    X["probabilities_a_1"] = y
    X["probabilities_a_0"] = y
    assert X.columns[2] == "probabilities_a_1"
    assert X.columns[3] == "probabilities_a_0"
    X.columns = list(map(str, X.columns))

    result = postpostprocess_output(X, name="a")
    assert result.columns[2] == "probabilities_a_0"
    assert result.columns[3] == "probabilities_a_1"


def test_clean_params():
    dummy_dict = dict(
        a=1,
        b=2,
        c=dict(
            d=3, _conditional="something", f=(dict(passthrough=True, hello="world"))
        ),
        _conditional="conditional",
        passthrough=True,
    )
    cleaned_dict = _clean_params(dummy_dict)

    assert cleaned_dict != dummy_dict, "Cleaned dict is the same as dummy dict."
    assert "passthrough" not in cleaned_dict, "Passthrough not removed."
    assert "_conditional" not in cleaned_dict, "_conditional not removed."
    assert (
        "_conditional" not in cleaned_dict["c"]
    ), "_conditional not removed from within dictionary."
    assert (
        "passthrough" not in cleaned_dict["c"]
    ), "passthrough not removed from within dictionary."
    assert (
        "passthrough" not in cleaned_dict["c"]["f"]
    ), "passthrough not removed from within dictionary."
    assert "a" in cleaned_dict, "missing value in cleaned dict"
    assert "b" in cleaned_dict, "missing value in cleaned dict"
    assert "hello" in cleaned_dict["c"]["f"], "missing value in cleaned dict"


def test_apply_function_batched():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
            "c": [1, 2, 3, 4, 5],
            "d": [1, 2, 3, 4, 5],
            "e": [1, 2, 3, 4, 5],
            "f": [1, 2, 3, 4, 5],
        }
    )
    assert apply_function_batched(df, np.log, 1).equals(np.log(df))
    assert apply_function_batched(df, np.log, 2).equals(np.log(df))
    assert apply_function_batched(df, np.log, 3).equals(np.log(df))
    assert apply_function_batched(df, np.log, 4).equals(np.log(df))


@pytest.fixture()
def sample_dataframe():
    return pd.DataFrame(
        {"col1": [1, 2, 3], "nocol2": ["a", "b", "c"], "nocol3": [True, False, True]}
    )


def test_get_column_names_all(sample_dataframe):
    expected = ["col1", "nocol2", "nocol3"]
    assert get_column_names("all", sample_dataframe).to_list() == expected


def test_get_column_names_exact_match(sample_dataframe):
    expected = ["nocol2"]
    assert get_column_names("nocol2", sample_dataframe) == expected


def test_get_column_names_prefix_match(sample_dataframe):
    expected = ["col1"]
    assert get_column_names("col*", sample_dataframe) == expected


def test_get_column_names_suffix_match(sample_dataframe):
    expected = ["nocol3"]
    assert get_column_names("*3", sample_dataframe) == expected


def test_get_column_names_no_match(sample_dataframe):
    expected = ["nonexistent_column"]
    assert get_column_names("nonexistent_column", sample_dataframe) == expected
