from __future__ import annotations

from copy import deepcopy
from typing import Callable, List, Optional, Union

import pandas as pd

from ..utils.list import wrap_in_list
from .base import Composite, Transformation, Transformations, TransformationsAlwaysList
from .concat import Concat, ResolutionStrategy
from .identity import Identity


class SelectColumns(Transformation):

    properties = Transformation.Properties(requires_past_X=False)

    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_in_list(columns)
        self.name = f"SelectColumns-{columns}"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns]


class DropColumns(Transformation):

    properties = Transformation.Properties(requires_past_X=False)

    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_in_list(columns)
        self.name = f"DropColumns-{columns}"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns)


class RenameColumns(Transformation):
    properties = Transformation.Properties(requires_past_X=False)

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

    properties = Composite.Properties()

    def __init__(self, transformations: Transformations) -> None:
        self.transformations = wrap_in_list(transformations)
        self.name = "PerColumnTransform-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in self.transformations
            ]
        )

    def before_fit(self, X: pd.DataFrame) -> None:
        self.transformations = [deepcopy(self.transformations) for _ in X.columns]

    def preprocess_X_primary(self, X: pd.DataFrame, index: int) -> pd.DataFrame:
        return X.iloc[:, index].to_frame()

    def postprocess_result_primary(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(results, axis=1)

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.transformations

    def clone(self, clone_child_transformations: Callable) -> PerColumnTransform:
        return PerColumnTransform(
            transformations=clone_child_transformations(self.transformations),
        )


class OnlyPredictions(Transformation):
    properties = Transformation.Properties(requires_past_X=False)

    def __init__(self) -> None:
        self.name = "OnlyPredictions"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(
            columns=[col for col in X.columns if not col.startswith("predictions_")]
        )


class OnlyProbabilities(Transformation):
    properties = Transformation.Properties(requires_past_X=False)

    def __init__(self) -> None:
        self.name = "OnlyProbabilities"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(
            columns=[col for col in X.columns if not col.startswith("probabilities_")]
        )


class SkipNA(Composite):

    """
    Skips rows with NaN values in the input data.
    Adds back the rows with NaN values after the transformations are applied.
    Enables transformations to be applied to data with missing values, without imputation.
    """

    properties = Composite.Properties()

    def __init__(self, transformations: Transformations) -> None:
        self.transformations = wrap_in_list(transformations)
        self.name = "SkipNA-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in self.transformations
            ]
        )

    def preprocess_X_primary(self, X: pd.DataFrame, index: int) -> pd.DataFrame:
        self.original_index = X.index.copy()
        self.isna = X.isna().any(axis=1)
        return X[~self.isna]

    def preprocess_y_primary(self, y: pd.Series) -> pd.Series:
        return y[~self.isna]

    def postprocess_result_primary(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        results = [result.reindex(self.original_index) for result in results]
        return pd.concat(results, axis=1)

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.transformations

    def clone(self, clone_child_transformations: Callable) -> SkipNA:
        return SkipNA(
            transformations=clone_child_transformations(self.transformations),
        )
