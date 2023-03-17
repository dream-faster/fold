from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..transformations.base import Transformations, TransformationsAlwaysList
from .base import Composite, T


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
