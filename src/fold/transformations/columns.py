from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..base import Transformation, fit_noop
from ..utils.checks import check_get_columns
from ..utils.list import wrap_in_list


class SelectColumns(Transformation):
    """
    Select a single or multiple columns.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns: List[str] = wrap_in_list(columns)
        self.name = f"SelectColumns-{columns}"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X[self.columns]

    fit = fit_noop
    update = fit


class DropColumns(Transformation):
    """
    Drops a single or multiple columns.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_in_list(columns)
        self.name = f"DropColumns-{columns}"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(columns=check_get_columns(self.columns, X))

    fit = fit_noop
    update = fit


class RenameColumns(Transformation):
    """
    Renames columns.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self, columns_mapper: dict) -> None:
        self.columns_mapper = columns_mapper
        self.name = "RenameColumns"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.rename(columns=self.columns_mapper)

    fit = fit_noop
    update = fit


class OnlyPredictions(Transformation):
    """
    Drops all columns except the output model(s)' predictions.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self) -> None:
        self.name = "OnlyPredictions"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(
            columns=[col for col in X.columns if not col.startswith("predictions_")]
        )

    fit = fit_noop
    update = fit


class OnlyProbabilities(Transformation):
    """
    Drops all columns except the output model(s)' probabilities.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self) -> None:
        self.name = "OnlyProbabilities"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(
            columns=[col for col in X.columns if not col.startswith("probabilities_")]
        )

    fit = fit_noop
    update = fit
