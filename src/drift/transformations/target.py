from typing import List

import pandas as pd

from ..transformations.base import Composite, Transformation, Transformations
from ..utils.list import wrap_in_list


class TransformTarget(Composite):
    def __init__(
        self, X_transformations: Transformations, y_transformation: Transformation
    ) -> None:
        self.X_transformations = [X_transformations]
        self.y_transformation = y_transformation
        self.name = "TransformTarget-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in wrap_in_list(X_transformations)
                + [y_transformation]
            ]
        )

    def preprocess_y(self, y: pd.Series) -> pd.Series:
        return self.y_transformation.fit_transform(X=y.to_frame(), y=None)

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return self.y_transformation.inverse_transform(results[0])

    def get_child_transformations(self) -> Transformations:
        return self.X_transformations
