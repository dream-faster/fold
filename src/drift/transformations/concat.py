from enum import Enum
from typing import List, Union

import pandas as pd

from .base import Composite, Transformations


class ResolutionStrategy(Enum):
    left = "left"
    right = "right"
    both = "both"


class Concat(Composite):
    def __init__(
        self,
        transformations: Transformations,
        if_duplicate_keep: Union[ResolutionStrategy, str] = ResolutionStrategy.both,
    ) -> None:
        self.transformations = transformations
        self.strategy = if_duplicate_keep
        self.name = "Concat-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in transformations
            ]
        )

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        results = pd.concat(results, axis=1)

        if self.strategy == ResolutionStrategy.left:
            return results.loc[:, ~results.columns.duplicated(keep="first")]
        elif self.strategy == ResolutionStrategy.right:
            return results.loc[:, ~results.columns.duplicated(keep="last")]
        elif self.strategy == ResolutionStrategy.both:
            return results
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def get_child_transformations(self) -> Transformations:
        return self.transformations

    def set_child_transformations(self, transformations: Transformations) -> None:
        self.transformations = transformations
