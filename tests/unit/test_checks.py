import pandas as pd

from fold.utils.checks import is_prediction


def test_is_prediction():
    X_one_column_prediction = pd.DataFrame(
        {
            "predictions_": [3, 4, 5],
        }
    )
    X_one_column_not_prediction = pd.DataFrame(
        {
            "not_predictions": [3, 4, 5],
        }
    )
    X_first_column_prediction = pd.DataFrame(
        {
            "predictions_": [3, 4, 5],
            "probabilities_": [3, 4, 5],
        }
    )
    X_first_column_prediction_other_not_probs = pd.DataFrame(
        {
            "predictions_": [3, 4, 5],
            "not_probabilities": [3, 4, 5],
        }
    )
    X_first_column_not_prediction = pd.DataFrame(
        {
            "not_predictions": [3, 4, 5],
            "b": [3, 4, 5],
        }
    )
    assert (
        is_prediction(X_first_column_prediction) is True
    ), "First column is prediction in X_first_column_prediction but is_prediction evaluates to False."
    assert (
        is_prediction(X_first_column_not_prediction) is False
    ), "First column is not prediction in X_first_column_not_prediction but is_prediction evaluates to True."

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
    pass


def test_get_prediction_column():
    pass


def test_get_probabilities_column():
    pass


def test_get_probabilities_column_name():
    pass


def test_get_prediction_column_name():
    pass


def test_is_X_available():
    pass


def test_is_columns_all():
    pass


def test_check_get_columns():
    pass
