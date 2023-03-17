from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..utils.list import wrap_in_list
from .base import Transformation, fit_noop


class SelectColumns(Transformation):
    properties = Transformation.Properties()

    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_in_list(columns)
        self.name = f"SelectColumns-{columns}"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X[self.columns]

    fit = fit_noop
    update = fit


class DropColumns(Transformation):
    properties = Transformation.Properties()

    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_in_list(columns)
        self.name = f"DropColumns-{columns}"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(columns=self.columns)

    fit = fit_noop
    update = fit


class RenameColumns(Transformation):
    properties = Transformation.Properties()

    def __init__(self, columns_mapper: dict) -> None:
        self.columns_mapper = columns_mapper
        self.name = "RenameColumns"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.rename(columns=self.columns_mapper)

    fit = fit_noop
    update = fit


class OnlyPredictions(Transformation):
    properties = Transformation.Properties()

    def __init__(self) -> None:
        self.name = "OnlyPredictions"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(
            columns=[col for col in X.columns if not col.startswith("predictions_")]
        )

    fit = fit_noop
    update = fit


class OnlyProbabilities(Transformation):
    properties = Transformation.Properties()

    def __init__(self) -> None:
        self.name = "OnlyProbabilities"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(
            columns=[col for col in X.columns if not col.startswith("probabilities_")]
        )

    fit = fit_noop
    update = fit
