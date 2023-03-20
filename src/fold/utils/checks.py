from typing import List

import pandas as pd


def is_prediction(input: pd.DataFrame) -> bool:
    if len(input.columns) == 1:
        return (
            input.columns[0].startswith("predictions_")
            if type(input.columns[0]) is str
            else False
        )
    else:
        is_predictions_col_present = input.columns[0].startswith("predictions_")
        return is_predictions_col_present and all(
            [col.startswith("probabilities_") for col in input.columns[1:]]
        )


def all_have_probabilities(results: List[pd.DataFrame]) -> bool:
    """
    Check if all the DataFrames have probabilities columns,
    or if their values are all NaN, indicating an empty DataFrame
    being passed around, as SkipNA filtered out all data.
    """
    return all(
        [
            any([True for col in df.columns if col.startswith("probabilities_")])
            or len(df.columns) == 0
            or (len(df.columns) == 1 and df.isna().sum()[0] == len(df))
            for df in results
        ]
    )


def get_prediction_column(input: pd.DataFrame) -> pd.Series:
    return input[get_prediction_column_name(input)].squeeze()


def get_prediction_column_name(input: pd.DataFrame) -> str:
    return [col for col in input.columns if col.startswith("predictions_")][0]


def is_X_available(X: pd.DataFrame) -> bool:
    """
    Check if X is available, or the input is univariate, without exogenous variables.
    """
    return X.shape[1] == 1 and X.columns[0] == "X_not_available"
