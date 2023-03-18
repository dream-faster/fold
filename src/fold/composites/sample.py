from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import pandas as pd

from fold.transformations.common import get_concatenated_names

from ..transformations.base import BlocksOrWrappable, TransformationsAlwaysList
from .base import Composite, T


class Sample(Composite):
    """
    Sample data with an imbalanced-learn sampler instance.
    """

    properties = Composite.Properties(
        primary_requires_predictions=False,
        primary_only_single_pipeline=True,
    )

    def __init__(
        self,
        sampler: Any,
        transformations: BlocksOrWrappable,
    ) -> None:
        self.sampler = sampler
        self.transformations = [transformations]
        self.name = f"Sample-{sampler.__class__.__name__}-{get_concatenated_names(self.transformations)}"

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        if fit:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            X_resampled.columns = X.columns
            X_resampled.index = X.index[: len(X_resampled)]
            if y is not None:
                y_resampled.name = y.name
                y_resampled.index = y.index[: len(y_resampled)]
            return X_resampled, y_resampled
        else:
            return X, y

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return results[0]

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.transformations

    def clone(self, clone_child_transformations: Callable) -> Sample:
        return Sample(
            sampler=self.sampler,
            transformations=clone_child_transformations(self.transformations),
        )
