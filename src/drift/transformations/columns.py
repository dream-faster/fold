from copy import deepcopy
from typing import Callable, List, Optional, Union

import pandas as pd

from ..utils.list import wrap_in_list
from .base import Composite, Transformation, Transformations
from .concat import Concat, ResolutionStrategy
from .identity import Identity


class SelectColumns(Transformation):
    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_in_list(columns)
        self.name = f"SelectColumns-{columns}"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns]


class DropColumns(Transformation):
    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_in_list(columns)
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
            [SelectColumns(columns)] + wrap_in_list(transformation),
            Identity(),
        ],
        if_duplicate_keep=ResolutionStrategy.left,
    )


class PerColumnTransform(Composite):
    def __init__(self, transformations: Transformations) -> None:
        self.transformations = transformations
        self.name = "PerColumnTransform-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in transformations
            ]
        )

    def before_fit(self, X: pd.DataFrame) -> None:
        self.transformations = [deepcopy(self.transformations) for _ in X.columns]

    def preprocess_X(
        self, X: pd.DataFrame, index: int, for_inference: bool
    ) -> pd.DataFrame:
        return X.iloc[:, index].to_frame()

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(results, axis=1)

    def get_child_transformations(self) -> Transformations:
        return self.transformations

    def clone(self, clone_child_transformations: Callable) -> Composite:
        return PerColumnTransform(
            transformations=clone_child_transformations(self.transformations),
        )
