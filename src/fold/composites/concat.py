from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional, Union

import pandas as pd

from ..base import Composite, Pipelines, Transformations
from ..transformations.columns import SelectColumns
from ..transformations.dev import Identity
from ..utils.list import (
    flatten,
    has_intersection,
    keep_only_duplicates,
    wrap_in_double_list_if_needed,
    wrap_in_list,
)
from .common import get_concatenated_names


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
    """
    Concatenates the results of multiple pipelines.
    """

    ResolutionStrategy = ResolutionStrategy
    properties = Composite.Properties()

    def __init__(
        self,
        pipelines: Pipelines,
        if_duplicate_keep: Union[ResolutionStrategy, str] = ResolutionStrategy.both,
    ) -> None:
        self.pipelines = pipelines
        self.if_duplicate_keep = ResolutionStrategy.from_str(if_duplicate_keep)
        self.name = "Concat-" + get_concatenated_names(pipelines)

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

    def get_child_transformations_primary(self) -> Pipelines:
        return self.pipelines

    def clone(self, clone_child_transformations: Callable) -> Concat:
        return Concat(
            pipelines=clone_child_transformations(self.pipelines),
            if_duplicate_keep=self.if_duplicate_keep,
        )


class Pipeline(Composite):
    """
    An optional wrapper that is equivalent to using a single array for the transformations.
    It executes the transformations in the order they are provided.
    """

    properties = Composite.Properties(primary_only_single_pipeline=True)

    def __init__(
        self,
        pipeline: "Pipeline",
    ) -> None:
        self.pipeline = wrap_in_double_list_if_needed(pipeline)
        self.name = "Pipeline-" + get_concatenated_names(pipeline)

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return results[0]

    def get_child_transformations_primary(self) -> Pipelines:
        return self.pipeline

    def clone(self, clone_child_transformations: Callable) -> Pipeline:
        return Pipeline(pipeline=clone_child_transformations(self.pipeline))


def TransformColumn(
    columns: Union[List[str], str], pipeline: Transformations
) -> Composite:
    """
    Transforms a single or multiple columns using the given pipeline.
    """
    return Concat(
        [
            [SelectColumns(columns)] + wrap_in_list(pipeline),
            Identity(),
        ],
        if_duplicate_keep=ResolutionStrategy.left,
    )
