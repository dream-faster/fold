# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import List

import pandas as pd

from fold.utils.dataframe import to_series


def is_prediction(input: pd.DataFrame) -> bool:
    if len(input.columns) == 1:
        return (
            input.columns[0].startswith("predictions_")
            if isinstance(input.columns[0], str)
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
    return to_series(input[get_prediction_column_name(input)])


def get_probabilities_columns(input: pd.DataFrame) -> pd.DataFrame:
    return input[get_probabilities_column_names(input)]


def get_probabilities_column_names(input: pd.DataFrame) -> List[str]:
    candidates = [col for col in input.columns if col.startswith("probabilities_")]
    if len(candidates) == 0:
        raise ValueError(f"Could not find any probabilities column in {input.columns}.")
    return candidates


def get_prediction_column_name(input: pd.DataFrame) -> str:
    candidates = [col for col in input.columns if col.startswith("predictions_")]
    if len(candidates) == 0:
        if len(input.columns) == 1:
            return input.columns[0]
        else:
            raise ValueError(f"Could not find a predictions column in {input.columns}.")
    return candidates[0]


def is_X_available(X: pd.DataFrame) -> bool:
    """
    Check if X is available, or the input is univariate, without exogenous variables.
    """
    return not (X.shape[1] == 1 and X.columns[0] == "X_not_available")


def is_columns_all(columns: List[str]):
    return columns[0] == "all"


def check_get_columns(columns: List[str], X: pd.DataFrame) -> List[str]:
    if is_columns_all(columns):
        return X.columns
    else:
        return columns
