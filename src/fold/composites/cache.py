# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

import os
from typing import Callable, List, Optional

import pandas as pd

from ..base import Artifact, Composite, Pipeline, Pipelines, get_concatenated_names
from ..transformations.dev import Identity
from ..utils.dataframe import ResolutionStrategy, concat_on_columns_with_duplicates
from ..utils.list import wrap_in_double_list_if_needed


class Cache(Composite):
    """
    Saves the results of the pipeline up until its position for the first time, to the given directory (in feathe format).
    If the file exists at the location, it loads it and skips execution of the wrapped pipeline.
    It only works during backtesting, and can not be used in live deployments.

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
        self,
        results: List[pd.DataFrame],
        y: Optional[pd.Series],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        metadata: Composite.Metadata = self.metadata  # type: ignore
        if metadata.inference is True:
            return results[0]
        if os.path.exists(self.path) and os.path.exists(
            _result_path(
                self.path,
                metadata.project_name,
                metadata.fold_index,
                metadata.target,
                fit,
            )
        ):
            return pd.read_feather(
                _result_path(
                    self.path,
                    metadata.project_name,
                    metadata.fold_index,
                    metadata.target,
                    fit,
                ),
                use_threads=False,
            ).set_index("index")
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            results[0].reset_index().to_feather(
                _result_path(
                    self.path,
                    metadata.project_name,
                    metadata.fold_index,
                    metadata.target,
                    fit,
                )
            )
            return results[0]

    def postprocess_artifacts_primary(
        self,
        primary_artifacts: List[Artifact],
        results: List[pd.DataFrame],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        metadata: Composite.Metadata = self.metadata  # type: ignore
        if metadata.inference is True:
            return primary_artifacts[0]
        if os.path.exists(self.path) and os.path.exists(
            _artifacts_path(
                self.path,
                metadata.project_name,
                metadata.fold_index,
                metadata.target,
                fit,
            )
        ):
            return pd.read_feather(
                _artifacts_path(
                    self.path,
                    metadata.project_name,
                    metadata.fold_index,
                    metadata.target,
                    fit,
                ),
                use_threads=False,
            ).set_index("index")
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            artifacts = concat_on_columns_with_duplicates(
                primary_artifacts,
                strategy=ResolutionStrategy.last,
            )
            artifacts.reset_index().to_feather(
                _artifacts_path(
                    self.path,
                    metadata.project_name,
                    metadata.fold_index,
                    metadata.target,
                    fit,
                )
            )
            return primary_artifacts[0]

    def get_children_primary(self) -> Pipelines:
        if self.metadata is None:
            return self.pipeline
        if self.metadata.inference is True:
            return self.pipeline
        if os.path.exists(self.path) and os.path.exists(
            _result_path(
                self.path,
                self.metadata.project_name,
                self.metadata.fold_index,
                self.metadata.target,
                False,
            )
        ):
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
        clone.id = self.id
        return clone


def __fit_to_str(fit: bool) -> str:
    return "fit" if fit else "predict"


def _result_path(
    path, project_name: str, fold_index: int, y_name: str, fit: bool
) -> str:
    return os.path.join(
        path,
        f"result_{project_name}_{y_name}_fold{str(fold_index)}_{__fit_to_str(fit)}.feather",
    )


def _artifacts_path(
    path, project_name: str, fold_index: int, y_name: str, fit: bool
) -> str:
    return os.path.join(
        path,
        f"artifacts_{project_name}_{y_name}_fold{str(fold_index)}_{__fit_to_str(fit)}.feather",
    )
