# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

import os
from typing import Callable, List, Optional

import pandas as pd

from ..base import Artifact, Composite, Pipeline, Pipelines, get_concatenated_names
from ..transformations.dev import Identity
from ..utils.list import wrap_in_double_list_if_needed


class Cache(Composite):
    """
    Saves the results of the pipeline up until its position for the first time, to the given directory (in parquet format).
    If the file exists at the location, it loads it and skips execution of the wrapped pipeline.

    Parameters
    ----------

    pipeline: Pipeline
        pipeline to execute if file at path doesn't exist.

    path: str
        path to the directory used for caching.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        path: str,
        name: Optional[str] = None,
    ) -> None:
        self.path = path
        self.pipeline: Pipelines = wrap_in_double_list_if_needed(pipeline)  # type: ignore
        self.name = name or "Cache-" + get_concatenated_names(self.pipeline)
        self.properties = Composite.Properties(
            primary_only_single_pipeline=True, artifacts_length_should_match=False
        )
        self.metadata = None

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        if os.path.exists(self.path) and os.path.exists(_result_path(self.path)):
            return pd.read_parquet(_result_path(self.path))
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            results[0].to_parquet(_result_path(self.path))
            return results[0]

    def postprocess_artifacts_primary(
        self,
        primary_artifacts: List[Artifact],
        results: List[pd.DataFrame],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        if os.path.exists(self.path) and os.path.exists(_artifacts_path(self.path)):
            return pd.read_parquet(_artifacts_path(self.path))
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            primary_artifacts[0].to_parquet(_artifacts_path(self.path))
            return primary_artifacts[0]

    def get_children_primary(self) -> Pipelines:
        if os.path.exists(self.path) and os.path.exists(_result_path(self.path)):
            return [Identity()]
        return self.pipeline

    def clone(self, clone_children: Callable) -> Cache:
        clone = Cache(
            pipeline=clone_children(self.pipeline),
            path=self.path,
        )
        clone.properties = self.properties
        clone.name = self.name
        clone.metadata = self.metadata
        return clone


def _result_path(path) -> str:
    return os.path.join(path, "result.parquet")


def _artifacts_path(path) -> str:
    return os.path.join(path, "artifacts.parquet")
