from typing import List, Optional, Union

import pandas as pd

from ..utils.list import wrap_into_list_if_needed
from .base import Composite, Transformation, Transformations
from .concat import Concat, ResolutionStrategy
from .identity import Identity


class SelectColumns(Transformation):
    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_into_list_if_needed(columns)
        self.name = f"SelectColumns-{columns}"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns]


class DropColumns(Transformation):
    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_into_list_if_needed(columns)
        self.name = f"DropColumns-{columns}"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns)


class RenameColumns(Transformation):
    def __init__(self, columns_mapper: dict) -> None:
        self.columns_mapper = columns_mapper
        self.name = "RenameColumns"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.rename(columns=self.columns_mapper)


def TransformColumn(
    columns: Union[List[str], str], transformation: Transformations
) -> Composite:
    return Concat(
        [
            [SelectColumns(columns)] + wrap_into_list_if_needed(transformation),
            Identity(),
        ],
        if_duplicate_keep=ResolutionStrategy.left,
    )
