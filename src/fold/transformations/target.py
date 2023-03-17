from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pandas as pd

from fold.transformations.common import get_concatenated_names
from fold.utils.checks import get_prediction_column_name

from ..transformations.base import (
    Composite,
    InvertibleTransformation,
    T,
    Transformations,
    TransformationsAlwaysList,
)


class TransformTarget(Composite):
    properties = Composite.Properties(
        primary_only_single_pipeline=True,
        secondary_only_single_pipeline=True,
        secondary_requires_predictions=True,
    )

    def __init__(
        self,
        X_transformations: Transformations,
        y_transformation: InvertibleTransformation,
    ) -> None:
        self.X_transformations = [X_transformations]
        self.y_transformation = y_transformation
        self.name = "TransformTarget-" + get_concatenated_names(
            self.X_transformations + [self.y_transformation]
        )

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        # TransformTarget's primary transformation transforms `y`, not `X`.
        if y is None:
            return (
                pd.DataFrame(),
                None,
            )  # at inference time, `y` will be None, and we don't need to use primary transformations at all, so we return a dummy DataFrame.
        else:
            return y.to_frame(), None

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        raise NotImplementedError

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        results_primary: List[pd.DataFrame],
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T]:
        return X, results_primary[0].squeeze()

    def postprocess_result_secondary(
        self,
        primary_results: List[pd.DataFrame],
        secondary_results: List[pd.DataFrame],
        y: Optional[pd.Series],
    ) -> pd.DataFrame:
        predictions = secondary_results[0]
        predictions[
            get_prediction_column_name(predictions)
        ] = self.y_transformation.inverse_transform(
            predictions[get_prediction_column_name(predictions)]
        )
        return predictions

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
