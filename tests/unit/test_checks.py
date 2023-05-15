import pandas as pd
import pytest

from fold.utils.checks import (
    all_have_probabilities,
    get_prediction_column,
    get_prediction_column_name,
    get_probabilities_column_names,
    get_probabilities_columns,
    is_prediction,
)

X_one_column_prediction = pd.DataFrame(
    {
        "predictions_a": [3, 4, 5],
    }
)
X_one_column_not_prediction = pd.DataFrame(
    {
        "not_predictions": [3, 4, 5],
    }
)
X_first_column_prediction_other_probability = pd.DataFrame(
    {
        "predictions_a": [3, 4, 5],
        "probabilities_a": [0.3, 0.4, 0.5],
    }
)
X_first_column_prediction_other_not_probs = pd.DataFrame(
    {
        "predictions_a": [3, 4, 5],
        "not_probabilities": [3, 4, 5],
    }
)
X_first_column_not_prediction_two_probabilities = pd.DataFrame(
    {
        "not_predictions": [3, 4, 5],
        "probabilities_a": [0.3, 0.4, 0.5],
        "probabilities_b": [0.3, 0.4, 0.5],
    }
)


def test_is_prediction():
    assert (
        is_prediction(X_first_column_prediction_other_probability) is True
    ), "First column is prediction in X_first_column_prediction_other_probability but is_prediction evaluates to False."
    assert (
        is_prediction(X_first_column_not_prediction_two_probabilities) is False
    ), "First column is not prediction in X_first_column_not_prediction_two_probabilities but is_prediction evaluates to True."

    assert (
        is_prediction(X_first_column_prediction_other_not_probs) is False
    ), "First column is prediction, but others are not probabilities in X_first_column_prediction_other_not_probs but is_prediction evaluates to True."

    assert (
        is_prediction(X_one_column_prediction) is True
    ), "First column is prediction in X_one_column_prediction but is_prediction evaluates to False."
    assert (
        is_prediction(X_one_column_not_prediction) is False
    ), "First column is not prediction X_one_column_not_prediction but is_prediction evaluates to True."


def test_all_have_probabilities():
    assert (
        all_have_probabilities(
            [
                X_first_column_prediction_other_probability,
                X_first_column_prediction_other_probability,
            ]
        )
        is True
    ), "All DataFrames have probabilities columns, but all_have_probabilities evaluates to False."
    assert (
        all_have_probabilities(
            [
                X_first_column_prediction_other_probability,
                X_first_column_prediction_other_not_probs,
            ]
        )
        is False
    ), "Not all DataFrames have probabilities columns, but all_have_probabilities evaluates to True."


def test_get_prediction_column():
    assert get_prediction_column(X_first_column_prediction_other_probability).equals(
        pd.Series([3, 4, 5], name="predictions_a")
    ), "get_prediction_column does not return the predictions column."

    with pytest.raises(ValueError):
        get_prediction_column(X_first_column_not_prediction_two_probabilities)


def test_get_probabilities_columns():
    assert get_probabilities_columns(
        X_first_column_not_prediction_two_probabilities
    ).equals(
        pd.DataFrame(
            {
                "probabilities_a": [0.3, 0.4, 0.5],
                "probabilities_b": [0.3, 0.4, 0.5],
            }
        )
    ), "get_probabilities_columns does not return the probabilities columns."

    with pytest.raises(ValueError):
        get_probabilities_columns(X_first_column_prediction_other_not_probs)


def test_get_probabilities_column_name():
    assert get_probabilities_column_names(
        X_first_column_not_prediction_two_probabilities
    ) == [
        "probabilities_a",
        "probabilities_b",
    ], "get_probabilities_column_names does not return the probabilities column names."
    assert get_probabilities_column_names(
        X_first_column_prediction_other_probability
    ) == [
        "probabilities_a"
    ], "get_probabilities_column_names does not return the probabilities column name."
    with pytest.raises(ValueError):
        get_probabilities_column_names(X_first_column_prediction_other_not_probs)


def test_get_prediction_column_name():
    assert (
        get_prediction_column_name(X_first_column_prediction_other_probability)
        == "predictions_a"
    ), "get_prediction_column_name does not return the prediction column name."
    with pytest.raises(ValueError):
        get_prediction_column_name(X_first_column_not_prediction_two_probabilities)
