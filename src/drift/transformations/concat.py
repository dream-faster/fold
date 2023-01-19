from enum import Enum
from typing import List, Union

import pandas as pd

from drift.utils.list import flatten, has_intersection, keep_only_duplicates

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
        self.if_duplicate_keep = if_duplicate_keep
        self.name = "Concat-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in transformations
            ]
        )

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        columns = flatten([result.columns.to_list() for result in results])
        duplicates = keep_only_duplicates(columns)

        if len(duplicates) > 0 or self.if_duplicate_keep != ResolutionStrategy.both:
            duplicate_columns = [
                result[duplicates]
                for result in results
                if has_intersection(result.columns.to_list(), duplicates)
            ]
            results = [result.drop(columns=duplicates) for result in results]
            if self.if_duplicate_keep == ResolutionStrategy.left:
                return pd.concat(results + [duplicate_columns[0]], axis=1)
            elif self.if_duplicate_keep == ResolutionStrategy.right:
                return pd.concat(results + [duplicate_columns[-1]], axis=1)
            else:
                raise ValueError(
                    f"ResolutionStrategy is not valid: {self.if_duplicate_keep}"
                )
        else:
            return pd.concat(results, axis=1)

    def get_child_transformations(self) -> Transformations:
        return self.transformations
