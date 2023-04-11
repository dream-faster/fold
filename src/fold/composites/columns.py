from __future__ import annotations

from copy import deepcopy
from typing import Callable, List, Optional, Tuple

import pandas as pd

from ..base import Composite, Pipeline, Pipelines, T
from ..utils.checks import all_have_probabilities
from ..utils.list import unique, wrap_in_double_list_if_needed
from .common import get_concatenated_names


class PerColumnEnsemble(Composite):
    """
    Train a pipeline for each column in the data.
    Ensemble their results.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline (list of Pipeline) to ensemble
    models_already_cloned: bool
        For internal use. It determines if the pipeline has been copied or not.

    Returns
    ----------
    X: pd.DataFrame
        Ensemble of outputs of passed in pipelines.
    y: pd.Series
        Target passed along.
    """

    properties = Composite.Properties()
    models_already_cloned = False

    def __init__(self, pipeline: Pipeline, models_already_cloned: bool = False) -> None:
        self.models: Pipelines = wrap_in_double_list_if_needed(pipeline)
        self.name = "PerColumnEnsemble-" + get_concatenated_names(self.models)
        self.models_already_cloned = models_already_cloned

    def before_fit(self, X: pd.DataFrame) -> None:
        if not self.models_already_cloned:
            self.models = [deepcopy(self.models) for _ in X.columns]
            self.models_already_cloned = True

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, pd.Series]:
        X = X.iloc[:, index].to_frame()
        return X, y

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return postprocess_results(results, self.name)

    def get_child_transformations_primary(self) -> Pipelines:
        return self.models

    def clone(self, clone_child_transformations: Callable) -> PerColumnEnsemble:
        return PerColumnEnsemble(
            pipeline=clone_child_transformations(self.models),
            models_already_cloned=self.models_already_cloned,
        )


class SkipNA(Composite):
    """
    Skips rows with NaN values in the input data.
    Adds back the rows with NaN values after the transformations are applied.
    Enables transformations to be applied to data with missing values, without imputation.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline (list of Pipeline) to ensemble

    Returns
    -------
    X: pd.DataFrame
        Original X that it has received.
    y: pd.Series
        Target passed along.
    """

    properties = Composite.Properties()

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = wrap_in_double_list_if_needed(pipeline)
        self.name = "SkipNA-" + get_concatenated_names(self.pipeline)

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        self.original_index = X.index.copy()
        self.isna = X.isna().any(axis=1)
        return X[~self.isna], y[~self.isna] if y is not None else None

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        results = [result.reindex(self.original_index) for result in results]
        return pd.concat(results, axis="columns")

    def get_child_transformations_primary(self) -> Pipelines:
        return self.pipeline

    def clone(self, clone_child_transformations: Callable) -> SkipNA:
        return SkipNA(
            pipeline=clone_child_transformations(self.pipeline),
        )


class PerColumnTransform(Composite):
    """
    Apply a single pipeline for each column, separately.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline that gets applied to each column

    Returns
    -------
    X: pd.DataFrame
        X with the pipeline applied to each column seperately.
    y: pd.Series
        Target passed along.
    """

    properties = Composite.Properties()

    def __init__(self, pipeline: Pipeline, pipeline_already_cloned=False) -> None:
        self.pipeline = wrap_in_double_list_if_needed(pipeline)
        self.name = "PerColumnTransform-" + get_concatenated_names(self.pipeline)
        self.pipeline_already_cloned = pipeline_already_cloned

    def before_fit(self, X: pd.DataFrame) -> None:
        if not self.pipeline_already_cloned:
            self.pipeline = [deepcopy(self.pipeline) for _ in X.columns]
            self.pipeline_already_cloned = True

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        return X.iloc[:, index].to_frame(), y

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return pd.concat(results, axis="columns")

    def get_child_transformations_primary(self) -> Pipelines:
        return self.pipeline

    def clone(self, clone_child_transformations: Callable) -> PerColumnTransform:
        return PerColumnTransform(
            pipeline=clone_child_transformations(self.pipeline),
            pipeline_already_cloned=self.pipeline_already_cloned,
        )


def postprocess_results(
    results: List[pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    if all_have_probabilities(results):
        return get_groupped_columns_classification(results, name)
    else:
        return get_groupped_columns_regression(results, name)


def get_groupped_columns_regression(
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
        )
        .mean(axis="columns")
        .rename(f"predictions_{name}")
        .to_frame()
    )


def get_groupped_columns_classification(
    results: List[pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    columns = results[0].columns.to_list()
    probabilities_columns = [col for col in columns if col.startswith("probabilities_")]
    classes = unique([line.split("_")[-1] for line in probabilities_columns])

    predictions = (
        pd.concat(
            [
                df[
                    [col for col in df.columns if col.startswith("predictions_")]
                ].squeeze()
                for df in results
            ],
            axis="columns",
        )
        .mean(axis="columns")
        .rename(f"predictions_{name}")
    )

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
    return pd.concat([predictions] + probabilities, axis="columns")
