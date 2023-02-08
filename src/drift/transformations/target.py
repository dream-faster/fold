from typing import Callable, List

import pandas as pd

from ..transformations.base import Composite, Transformation, Transformations
from ..utils.list import wrap_in_list


class TransformTarget(Composite):
    def __init__(
        self, X_transformations: Transformations, y_transformation: Transformation
    ) -> None:
        self.X_transformations = wrap_in_list(X_transformations)
        self.y_transformation = y_transformation
        self.name = "TransformTarget-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in self.X_transformations + [y_transformation]
            ]
        )

    def preprocess_y(self, y: pd.Series) -> pd.Series:
        y_frame = y
        if isinstance(y, pd.Series):
            y_frame = y_frame.to_frame()
        self.y_transformation.fit(X=y_frame, y=None)
        return self.y_transformation.transform(X=y_frame).squeeze()

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return self.y_transformation.inverse_transform(results[0])

    def get_child_transformations(self) -> Transformations:
        return self.X_transformations

    def clone(self, clone_child_transformations: Callable) -> Composite:
        return TransformTarget(
            X_transformations=clone_child_transformations(self.X_transformations),
            y_transformation=clone_child_transformations(self.y_transformation),
        )
