from typing import List, Optional

import pandas as pd

from .base import Composite, Transformation


class Concat(Transformation, Composite):
    def __init__(self, transformations: List[Transformation]) -> None:
        # TODO: figure out a merge strategy if there are overlapping columns
        self.transformations = transformations
        self.name = "Concat-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in transformations
            ]
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        # to be done in the training loop with get_child_transformations()
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [transformation.transform(X) for transformation in self.transformations],
            axis=1,
        )

    def get_child_transformations(self) -> Optional[List[Transformation]]:
        return self.transformations

    def set_child_transformations(self, transformations: List[Transformation]) -> None:
        self.transformations = transformations
