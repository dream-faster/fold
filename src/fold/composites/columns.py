# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from copy import deepcopy
from typing import Callable, List, Optional, Tuple

import pandas as pd

from fold.base.classes import Artifact

from ..base import Composite, Pipeline, Pipelines, T, get_concatenated_names
from ..utils.checks import all_have_probabilities
from ..utils.list import unique, wrap_in_double_list_if_needed


class EnsembleEachColumn(Composite):
    """
    Train a pipeline for each column in the data, then ensemble their results.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline that get applied to every column, independently, their results then averaged.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.composites import EnsembleEachColumn
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from fold.utils.tests import generate_sine_wave_data
        >>> X, y  = generate_sine_wave_data()
        >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
        >>> pipeline = EnsembleEachColumn(RandomForestRegressor())
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)

    """

    pipelines_already_cloned = False

    def __init__(self, pipeline: Pipeline, name: Optional[str] = None) -> None:
        self.pipelines: Pipelines = wrap_in_double_list_if_needed(pipeline)  # type: ignore
        self.name = name or "PerColumnEnsemble-" + get_concatenated_names(
            self.pipelines
        )
        self.properties = Composite.Properties()
        self.metadata = None

    @classmethod
    def from_cloned_instance(
        cls, pipeline: Pipeline, pipelines_already_cloned: bool, name: Optional[str]
    ) -> EnsembleEachColumn:
        instance = cls(pipeline=pipeline)
        instance.pipelines_already_cloned = pipelines_already_cloned
        instance.name = name
        return instance

    def before_fit(self, X: pd.DataFrame) -> None:
        if not self.pipelines_already_cloned:
            self.pipelines = [deepcopy(self.pipelines) for _ in X.columns]
            self.pipelines_already_cloned = True

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        X = X.iloc[:, index].to_frame()
        return X, y, artifact

    def postprocess_result_primary(
        self,
        results: List[pd.DataFrame],
        y: Optional[pd.Series],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return average_results(results, self.name)

    def get_children_primary(self) -> Pipelines:
        return self.pipelines

    def clone(self, clone_children: Callable) -> EnsembleEachColumn:
        clone = EnsembleEachColumn.from_cloned_instance(
            pipeline=clone_children(self.pipelines),
            pipelines_already_cloned=self.pipelines_already_cloned,
            name=self.name,
        )
        clone.properties = self.properties
        clone.name = self.name
        clone.metadata = self.metadata
        clone.id = self.id
        return clone


class TransformEachColumn(Composite):
    """
    Apply a single pipeline to each column, separately.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline that gets applied to each column

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.composites import TransformEachColumn
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from fold.utils.tests import generate_sine_wave_data
        >>> X, y  = generate_sine_wave_data()
        >>> X["sine_plus_1"] = X["sine"] + 1.0
        >>> X.head()
                               sine  sine_plus_1
        2021-12-31 07:20:00  0.0000       1.0000
        2021-12-31 07:21:00  0.0126       1.0126
        2021-12-31 07:22:00  0.0251       1.0251
        2021-12-31 07:23:00  0.0377       1.0377
        2021-12-31 07:24:00  0.0502       1.0502
        >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
        >>> pipeline = TransformEachColumn(lambda x: x + 1.0)
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
        >>> preds.head()
                               sine  sine_plus_1
        2021-12-31 15:40:00  1.0000       2.0000
        2021-12-31 15:41:00  1.0126       2.0126
        2021-12-31 15:42:00  1.0251       2.0251
        2021-12-31 15:43:00  1.0377       2.0377
        2021-12-31 15:44:00  1.0502       2.0502
    """

    pipeline_already_cloned = False

    def __init__(self, pipeline: Pipeline, name: Optional[str] = None) -> None:
        self.pipeline = wrap_in_double_list_if_needed(pipeline)
        self.name = name or "PerColumnTransform-" + get_concatenated_names(
            self.pipeline
        )
        self.properties = Composite.Properties()
        self.metadata = None

    @classmethod
    def from_cloned_instance(
        cls, pipeline: Pipeline, pipeline_already_cloned: bool
    ) -> TransformEachColumn:
        instance = cls(pipeline=pipeline)
        instance.pipeline_already_cloned = pipeline_already_cloned
        return instance

    def before_fit(self, X: pd.DataFrame) -> None:
        if not self.pipeline_already_cloned:
            self.pipeline = [deepcopy(self.pipeline) for _ in X.columns]
            self.pipeline_already_cloned = True

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        return (X.iloc[:, index].to_frame(), y, artifact)

    def postprocess_result_primary(
        self,
        results: List[pd.DataFrame],
        y: Optional[pd.Series],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return pd.concat(results, copy=False, axis="columns")

    def get_children_primary(self) -> Pipelines:
        return self.pipeline

    def clone(self, clone_children: Callable) -> TransformEachColumn:
        clone = TransformEachColumn.from_cloned_instance(
            pipeline=clone_children(self.pipeline),
            pipeline_already_cloned=self.pipeline_already_cloned,
        )
        clone.properties = self.properties
        clone.name = self.name
        clone.metadata = self.metadata
        clone.id = self.id
        return clone


class SkipNA(Composite):
    """
    Skips rows with NaN values in the input data.
    In the output, rows with NaNs are returned as is, all other rows transformed.

    Warning:
    This seriously challenges the continuity of the data, which is very important for traditional time series models.
    Use with caution, and only with tabular ML models.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline to run without NA values.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> import numpy as np
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.composites import ModelResiduals
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from imblearn.under_sampling import RandomUnderSampler
        >>> from fold.utils.tests import generate_zeros_and_ones
        >>> X, y  = generate_zeros_and_ones()
        >>> X[1:100] = np.nan
        >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
        >>> pipeline = SkipNA(
        ...     pipeline=RandomForestClassifier(),
        ... )
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)

    """

    original_index: Optional[pd.Index] = None

    def __init__(self, pipeline: Pipeline, name: Optional[str] = None) -> None:
        self.pipeline = wrap_in_double_list_if_needed(pipeline)
        self.name = name or "SkipNA-" + get_concatenated_names(self.pipeline)
        self.properties = Composite.Properties(primary_only_single_pipeline=True)
        self.metadata = None
        self.isna = None

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        self.original_index = X.index.copy()
        self.isna = X.isna().any(axis=1)
        return (
            X[~self.isna],
            y[~self.isna] if y is not None else None,
            artifact[~self.isna],
        )

    def postprocess_result_primary(
        self,
        results: List[pd.DataFrame],
        y: Optional[pd.Series],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return results[0].reindex(self.original_index)

    def postprocess_artifacts_primary(
        self,
        primary_artifacts: List[Artifact],
        results: List[pd.DataFrame],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return primary_artifacts[0].reindex(self.original_index)

    def get_children_primary(self) -> Pipelines:
        return self.pipeline

    def clone(self, clone_children: Callable) -> SkipNA:
        clone = SkipNA(
            pipeline=clone_children(self.pipeline),
        )
        clone.properties = self.properties
        clone.original_index = self.original_index
        clone.name = self.name
        clone.metadata = self.metadata
        clone.isna = self.isna
        clone.id = self.id
        return clone


def average_results(
    results: List[pd.DataFrame],
    name: str,
    reconstruct_predictions_from_probabilities: bool = False,
) -> pd.DataFrame:
    if all_have_probabilities(results):
        return _average_results_classification(
            results, name, reconstruct_predictions_from_probabilities
        )
    else:
        return _average_results_regression(results, name)


def _average_results_regression(
    results: List[pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    return (
        pd.concat(
            [
                df[
                    [col for col in df.columns if col.startswith("predictions_")]
                ].squeeze()
                for df in results
            ],
            axis="columns",
            copy=False,
        )
        .mean(axis="columns")
        .rename(f"predictions_{name}")
        .to_frame()
    )


def _average_results_classification(
    results: List[pd.DataFrame],
    name: str,
    reconstruct_predictions_from_probabilities: bool = False,
) -> pd.DataFrame:
    columns = results[0].columns.to_list()
    probabilities_columns = [col for col in columns if col.startswith("probabilities_")]
    classes = unique([line.split("_")[-1] for line in probabilities_columns])

    probabilities = [
        (
            pd.concat(
                [
                    df[
                        [
                            col
                            for col in df.columns
                            if col.startswith("probabilities_")
                            and col.split("_")[-1] == selected_class
                        ]
                    ].squeeze()
                    for df in results
                ],
                axis="columns",
            )
            .mean(axis="columns")
            .rename(f"probabilities_{name}_{selected_class}")
        )
        for selected_class in classes
    ]
    if reconstruct_predictions_from_probabilities:
        above_05 = probabilities[1] >= 0.5
        predictions = above_05.astype(int).rename(f"predictions_{name}")

    else:
        predictions = (
            pd.concat(
                [
                    df[
                        [col for col in df.columns if col.startswith("predictions_")]
                    ].squeeze()
                    for df in results
                ],
                axis="columns",
                copy=False,
            )
            .mean(axis="columns")
            .rename(f"predictions_{name}")
        )

    return pd.concat([predictions] + probabilities, copy=True, axis="columns")
