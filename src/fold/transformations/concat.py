from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional, Union

import pandas as pd

from ..utils.list import flatten, has_intersection, keep_only_duplicates
from .base import Composite, Transformations, TransformationsAlwaysList


class ResolutionStrategy(Enum):
    left = "left"
    right = "right"
    both = "both"

    @staticmethod
    def from_str(value: Union[str, ResolutionStrategy]) -> ResolutionStrategy:
        if isinstance(value, ResolutionStrategy):
            return value
        for strategy in ResolutionStrategy:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown ResolutionStrategy: {value}")


class Concat(Composite):
    properties = Composite.Properties()

    def __init__(
        self,
        transformations: Transformations,
        if_duplicate_keep: Union[ResolutionStrategy, str] = ResolutionStrategy.both,
    ) -> None:
        self.transformations = transformations
        self.if_duplicate_keep = ResolutionStrategy.from_str(if_duplicate_keep)
        self.name = "Concat-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in transformations
            ]
        )

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
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
                return pd.concat(results + [duplicate_columns[0]], axis="columns")
            elif self.if_duplicate_keep == ResolutionStrategy.right:
                return pd.concat(results + [duplicate_columns[-1]], axis="columns")
            else:
                raise ValueError(
                    f"ResolutionStrategy is not valid: {self.if_duplicate_keep}"
                )
        else:
            return pd.concat(results, axis="columns")

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.transformations

    def clone(self, clone_child_transformations: Callable) -> Concat:
        return Concat(
            transformations=clone_child_transformations(self.transformations),
            if_duplicate_keep=self.if_duplicate_keep,
        )


class Pipeline(Composite):
    properties = Composite.Properties(primary_only_single_pipeline=True)

    def __init__(
        self,
        transformations: Transformations,
    ) -> None:
        self.transformations = transformations
        self.name = "Pipeline-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in transformations
            ]
        )

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return results[0]

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return [self.transformations]

    def clone(self, clone_child_transformations: Callable) -> Pipeline:
        return Pipeline(
            transformations=clone_child_transformations(self.transformations)
        )
