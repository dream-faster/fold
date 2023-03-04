from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd
from fold.transformations.common import get_concatenated_names

from ..transformations.base import Composite, Transformations, TransformationsAlwaysList


class TransformTarget(Composite):

    properties = Composite.Properties(
        primary_only_single_pipeline=True,
        secondary_only_single_pipeline=True,
    )

    def __init__(
        self, X_transformations: Transformations, y_transformation: Transformations
    ) -> None:
        self.X_transformations = [X_transformations]
        self.y_transformation = y_transformation
        self.name = "TransformTarget-" + get_concatenated_names(
            self.X_transformations + [self.y_transformation]
        )

    def preprocess_X_primary(
        self, X: pd.DataFrame, index: int, y: Optional[pd.Series]
    ) -> pd.DataFrame:
        # TransformTarget's primary transformation transforms `y`, not `X`.
        if y is None:
            return (
                pd.DataFrame()
            )  # at inference time, `y` will be None, and we don't need to use primary transformations at all, so we return a dummy DataFrame.
        else:
            return y.to_frame()

    def preprocess_y_primary(self, y: pd.Series) -> pd.Series:
        # TransformTarget's primary transformation (that transforms `y`) needs to be "unsupervised", it won't have access to `y`.
        return None

    def postprocess_result_primary(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError

    def preprocess_y_secondary(
        self, y: pd.Series, results_primary: List[pd.DataFrame]
    ) -> pd.Series:
        return results_primary[0].squeeze()

    def postprocess_result_secondary(
        self, primary_results: List[pd.DataFrame], secondary_results: List[pd.DataFrame]
    ) -> pd.DataFrame:
        return secondary_results[0]

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return [self.y_transformation]

    def get_child_transformations_secondary(
        self,
    ) -> Optional[TransformationsAlwaysList]:
        return self.X_transformations

    def clone(self, clone_child_transformations: Callable) -> TransformTarget:
        return TransformTarget(
            X_transformations=clone_child_transformations(self.X_transformations),
            y_transformation=clone_child_transformations(self.y_transformation),
        )
