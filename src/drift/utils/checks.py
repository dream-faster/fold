import pandas as pd


def is_prediction(input: pd.DataFrame) -> bool:
    if len(input.columns) == 1:
        return input.columns[0].startswith("predictions_")
    else:
        is_predictions_col_present = input.columns[0].startswith("predictions_")
        return is_predictions_col_present and all(
            [col.startswith("probabilities_") for col in input.columns]
        )


def get_prediction_column(input: pd.DataFrame) -> pd.Series:
    return input[
        [col for col in input.columns if col.startswith("predictions_")][0]
    ].squeeze()
