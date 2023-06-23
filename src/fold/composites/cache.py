# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

import os
from typing import Callable, List, Optional

import pandas as pd

from ..base import Artifact, Composite, Pipeline, Pipelines, get_concatenated_names
from ..utils.dataframe import concat_on_columns_with_duplicates
from ..utils.list import wrap_in_double_list_if_needed


class Cache(Composite):
    """
    Saves the results of the pipeline up until its position for the first time, to the given path (in parquet format).
    If the file exists at the location, it loads it and skips execution of all Blocks.

    Parameters
    ----------

    pipeline: Pipeline
        pipeline to execute if file at path doesn't exist.

    path: str
        path to the parquet file its using for caching.
    """

    properties = Composite.Properties()

    def __init__(
        self,
        pipeline: Pipeline,
        path: str,
        name: Optional[str] = None,
    ) -> None:
        self.path = path
        self.pipeline: Pipelines = wrap_in_double_list_if_needed(pipeline)  # type: ignore
        self.name = name or "Cache-" + get_concatenated_names(pipelines)

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        if os.path.exists(self.path):
            return pd.read_parquet(self.path)
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            results[0].to_parquet(self.path)
            return results[0]

    def postprocess_artifacts_primary(
        self,
        primary_artifacts: List[Artifact],
        results: List[pd.DataFrame],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return concat_on_columns_with_duplicates(
            primary_artifacts, self.if_duplicate_keep
        )

    def get_children_primary(self) -> Pipelines:
        # TODOD: return empty array if path exists
        if os.path.exists(self.path):
            return []
        return self.pipelines

    def clone(self, clone_children: Callable) -> Cache:
        clone = Cache(
            pipeline=clone_children(self.pipeline),
            path=self.path,
        )
        clone.properties = self.properties
        clone.name = self.name
        return clone
