from __future__ import annotations

from copy import deepcopy
from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..transformations.base import Transformations, TransformationsAlwaysList
from ..transformations.common import get_concatenated_names
from ..utils.list import wrap_in_list
from .base import Composite, T


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
