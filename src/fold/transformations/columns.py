from __future__ import annotations

from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from fold.transformations.common import get_concatenated_names

from ..utils.list import wrap_in_list
from .base import (
    Composite,
    T,
    Transformation,
    Transformations,
    TransformationsAlwaysList,
    fit_noop,
)
from .concat import Concat, ResolutionStrategy
from .identity import Identity


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

    def __init__(
        self, transformations: Transformations, transformations_already_cloned=False
    ) -> None:
        self.transformations = wrap_in_list(transformations)
        self.name = "PerColumnTransform-" + get_concatenated_names(self.transformations)
        self.transformations_already_cloned = transformations_already_cloned

    def before_fit(self, X: pd.DataFrame) -> None:
        if not self.transformations_already_cloned:
            self.transformations = [deepcopy(self.transformations) for _ in X.columns]
            self.transformations_already_cloned = True

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        return X.iloc[:, index].to_frame(), y

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return pd.concat(results, axis="columns")

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.transformations

    def clone(self, clone_child_transformations: Callable) -> PerColumnTransform:
        return PerColumnTransform(
            transformations=clone_child_transformations(self.transformations),
            transformations_already_cloned=self.transformations_already_cloned,
        )


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


class SkipNA(Composite):

    """
    Skips rows with NaN values in the input data.
    Adds back the rows with NaN values after the transformations are applied.
    Enables transformations to be applied to data with missing values, without imputation.
    """

    properties = Composite.Properties()

    def __init__(self, transformations: Transformations) -> None:
        self.transformations = [transformations]
        self.name = "SkipNA-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in self.transformations
            ]
        )

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        self.original_index = X.index.copy()
        self.isna = X.isna().any(axis=1)
        return X[~self.isna], y[~self.isna] if y is not None else None

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        results = [result.reindex(self.original_index) for result in results]
        return pd.concat(results, axis="columns")

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.transformations

    def clone(self, clone_child_transformations: Callable) -> SkipNA:
        return SkipNA(
            transformations=clone_child_transformations(self.transformations),
        )
