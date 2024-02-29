# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

import logging
from typing import TypeVar

import pandas as pd
from finml_utils.dataframes import concat_on_columns, concat_on_index, trim_initial_nans
from tqdm import tqdm
from fold.base.classes import BlockMetadata

from fold.loop.backend.sequential import NoBackend

from ..base import (
    Artifact,
    Backend,
    Composite,
    Optimizer,
    Pipeline,
    Sampler,
    TrainedPipeline,
    TrainedPipelines,
    Transformation,
    X,
)
from ..models.base import Model
from ..splitters import Fold, get_splits
from ..utils.checks import is_prediction
from ..utils.list import unpack_list_of_tuples
from .process.process_minibatch import _process_minibatch_transformation
from .types import Stage
from .utils import (
    _cut_to_backtesting_window,
    _cut_to_train_window,
    _extract_trained_pipelines,
    _set_metadata,
    deepcopy_pipelines,
    replace_with,
)

logger = logging.getLogger("fold:loop")
DEBUG_MULTI_PROCESSING = False

T = TypeVar(
    "T",
    bound=Transformation
    | Composite
    | Optimizer
    | Sampler
    | list[Transformation | Optimizer | Sampler | Composite],
)


def __post_checks(
    pipeline: T, X: pd.DataFrame, artifacts: Artifact
) -> tuple[T, X, Artifact]:
    assert X.shape[0] == artifacts.shape[0]
    assert X.index.equals(artifacts.index)
    return pipeline, X, artifacts


def recursively_transform(
    X: X,
    y: pd.Series | None,
    artifacts: Artifact,
    transformations: T,
    stage: Stage,
    backend: Backend,
    tqdm: tqdm | None = None,
) -> tuple[T, X, Artifact]:
    """
    The main function to transform (and fit or update) a pipline of transformations.
    """
    logger.debug(f"Processing {transformations.__class__.__name__} with stage {stage}")

    if tqdm is not None and hasattr(transformations, "name"):
        tqdm.set_description(f"Processing: {transformations.name}")

    if isinstance(transformations, list | tuple):
        processed_transformations = []
        for transformation in transformations:
            processed_transformation, X, artifacts = recursively_transform(
                X,
                y,
                artifacts,
                transformation,
                stage,
                backend,
                tqdm,
            )
            processed_transformations.append(processed_transformation)
        return __post_checks(processed_transformations, X, artifacts)

    if isinstance(transformations, Composite):
        return __post_checks(
            *_process_composite(
                transformations,
                X,
                y,
                artifacts,
                stage,
                backend,
                tqdm,
            )
        )
    if isinstance(transformations, Optimizer):
        return __post_checks(
            *_process_optimizer(
                transformations,
                X,
                y,
                artifacts,
                stage,
                backend,
                tqdm,
            )
        )

    if isinstance(transformations, Sampler):
        return __post_checks(
            *_process_sampler(
                transformations,
                X,
                y,
                artifacts,
                stage,
                backend,
                tqdm,
            )
        )

    if isinstance(transformations, Model | Transformation):
        return __post_checks(
            *_process_minibatch_transformation(
                transformations,
                X,
                y,
                artifacts,
                stage,
            )
        )

    raise ValueError(
        f"{transformations} is not a Fold Transformation, but of type"
        f" {type(transformations)}"
    )


def _process_composite(
    composite: Composite,
    X: pd.DataFrame,
    y: pd.Series | None,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    tqdm: tqdm | None = None,
) -> tuple[Composite, X, Artifact]:
    composite.before_fit(X)
    primary_transformations = composite.get_children_primary(only_traversal=False)

    (
        primary_transformations,
        results_primary,
        y_primary,
        artifacts_primary,
    ) = unpack_list_of_tuples(
        backend.process_child_transformations(
            __process_primary_child_transform,
            enumerate(primary_transformations),
            composite,
            X,
            y,
            artifacts,
            stage,
            backend,
            None,
            tqdm,
        )
    )
    if composite.properties.artifacts_length_should_match:
        assert all(
            r.shape[0] == a.shape[0]
            for r, a in zip(results_primary, artifacts_primary, strict=True)
        ), ValueError("Artifacts shape doesn't match result's length.")
    composite = composite.clone(replace_with(primary_transformations))

    if composite.properties.primary_only_single_pipeline:
        assert len(results_primary) == 1, ValueError(
            "Expected single output from primary transformations, got"
            f" {len(results_primary)} instead."
        )
    if composite.properties.primary_requires_predictions:
        assert is_prediction(results_primary[0]), ValueError(
            "Expected predictions from primary transformations, but got something else."
        )

    secondary_transformations = composite.get_children_secondary()

    original_results_primary = results_primary
    results_primary = composite.postprocess_result_primary(
        results=results_primary,
        y=y_primary[0],
        original_artifact=artifacts,
        fit=stage.is_fit(),
    )
    artifacts_primary = composite.postprocess_artifacts_primary(
        primary_artifacts=artifacts_primary,
        results=original_results_primary,
        fit=stage.is_fit(),
        original_artifact=artifacts,
    )
    if composite.properties.artifacts_length_should_match:
        assert artifacts_primary.shape[0] == results_primary.shape[0], ValueError(
            f"Artifacts shape doesn't match result's length after {composite.__class__.__name__}.postprocess_artifacts_primary() was called"
        )
    if secondary_transformations is None:
        return (
            composite,
            results_primary,
            artifacts_primary,
        )

    (
        secondary_transformations,
        results_secondary,
        artifacts_secondary,
    ) = unpack_list_of_tuples(
        backend.process_child_transformations(
            __process_secondary_child_transform,
            enumerate(secondary_transformations),
            composite,
            X,
            y,
            artifacts,
            stage,
            backend,
            results_primary,
            tqdm,
        )
    )
    composite = composite.clone(replace_with(secondary_transformations))

    if composite.properties.secondary_only_single_pipeline:
        assert len(results_secondary) == 1, ValueError(
            "Expected single output from secondary transformations, got"
            f" {len(results_secondary)} instead."
        )
    if composite.properties.secondary_requires_predictions:
        assert is_prediction(results_secondary[0]), ValueError(
            "Expected predictions from secondary transformations, but got"
            " something else."
        )

    return (
        composite,
        composite.postprocess_result_secondary(
            results_primary,
            results_secondary,
            y,
            in_sample=stage == Stage.inital_fit,
        ),
        composite.postprocess_artifacts_secondary(
            artifacts_primary, artifacts_secondary, artifacts
        ),
    )


def _process_sampler(
    sampler: Sampler,
    X: pd.DataFrame,
    y: pd.Series | None,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    tqdm: tqdm | None = None,
) -> tuple[Composite, X, Artifact]:
    primary_transformations = sampler.get_children_primary(only_traversal=False)

    (
        primary_transformations,
        primary_results,
        y_primary,
        primary_artifacts,
    ) = unpack_list_of_tuples(
        backend.process_child_transformations(
            __process_primary_child_transform,
            enumerate(primary_transformations),
            sampler,
            X,
            y,
            artifacts,
            stage,
            backend,
            None,
            tqdm,
        )
    )
    sampler = sampler.clone(replace_with(primary_transformations))

    assert len(primary_results) == 1, ValueError(
        "Expected single output from primary transformations, got"
        f" {len(primary_results)} instead."
    )

    if stage is Stage.inital_fit:
        secondary_transformations = sampler.get_children_primary(only_traversal=False)
        (
            secondary_transformations,
            primary_results,
            primary_y,
            primary_artifacts,
        ) = unpack_list_of_tuples(
            backend.process_child_transformations(
                __process_primary_child_transform,
                enumerate(secondary_transformations),
                sampler,
                X,
                y,
                artifacts,
                Stage.infer,
                backend,
                None,
                tqdm,
            )
        )

    primary_results = primary_results[0]
    primary_artifacts = primary_artifacts[0]
    assert primary_artifacts.shape[0] == primary_results.shape[0], ValueError(
        f"Artifacts shape doesn't match result's length after {sampler.__class__.__name__}.postprocess_artifacts_primary() was called"
    )
    return sampler, primary_results, primary_artifacts


def _process_optimizer(
    optimizer: Optimizer,
    X: pd.DataFrame,
    y: pd.Series | None,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    tqdm: tqdm | None = None,
) -> tuple[Pipeline, X, Artifact]:
    if tqdm is not None:
        tqdm.set_description(f"Processing: {optimizer.name}")
    optimized_pipeline = optimizer.get_optimized_pipeline()
    artifact = None
    if optimized_pipeline is None:
        while True:
            candidates = optimizer.get_candidates(only_traversal=False)
            if len(candidates) == 0:
                break

            _, results, candidate_artifacts = unpack_list_of_tuples(
                backend.process_child_transformations(
                    __process_candidates,
                    enumerate(candidates),
                    optimizer,
                    X,
                    y,
                    artifacts,
                    stage,
                    optimizer.backend if optimizer.backend is not None else backend,
                    None,
                    None,
                )
            )
            results = [trim_initial_nans(result) for result in results]
            artifact = optimizer.process_candidate_results(
                results,
                y=y.loc[results[0].index],
                artifacts=candidate_artifacts,
            )

    optimized_pipeline = optimizer.get_optimized_pipeline()
    processed_optimized_pipeline, X, artifact = recursively_transform(
        X,
        y,
        concat_on_columns([artifact, artifacts]),
        optimized_pipeline,
        stage,
        backend,
    )
    return optimizer, X, artifact


def __process_candidates(
    optimizer: Optimizer,
    index: int,
    child_transform: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | None,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: list[pd.DataFrame] | None,
    tqdm: tqdm | None = None,
) -> tuple[Pipeline, X, Artifact]:
    splits = get_splits(
        index=y.index,
        splitter=optimizer.splitter,
        gap_before=optimizer.metadata.project_hyperparameters["forward_horizon"]
        if optimizer.metadata.project_hyperparameters
        else 0,
        gap_after=0,
    )
    (processed_idx, processed_pipelines, _, _) = unpack_list_of_tuples(
        backend.train_pipeline(
            _train_on_window,
            pipeline=child_transform,
            X=X,
            y=y,
            artifact=artifacts,
            splits=splits,
            backend=NoBackend(),
            project_name=f"HPO-{optimizer.name}",
            project_hyperparameters=None,
            preprocessing_max_memory_size=optimizer.metadata.preprocessing_max_memory_size,
            silent=True,
        )
    )
    trained_pipelines = _extract_trained_pipelines(processed_idx, processed_pipelines)

    results, artifacts = unpack_list_of_tuples(
        backend.backtest_pipeline(
            _backtest_on_window,
            trained_pipelines,
            splits,
            X,
            y,
            artifacts,
            backend=NoBackend(),
            mutate=False,
            silent=True,
        )
    )
    return (
        trained_pipelines,
        trim_initial_nans(concat_on_index(results)),
        concat_on_index(artifacts),
    )


def __process_primary_child_transform(
    composite: Composite | Sampler,
    index: int,
    child_transform: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | None,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: list[pd.DataFrame] | None,
    tqdm: tqdm | None = None,
) -> tuple[Pipeline, X, pd.Series | None, Artifact]:
    X, y, artifacts = composite.preprocess_primary(
        X=X, index=index, y=y, artifact=artifacts, fit=stage.is_fit()
    )
    transformations, X, artifacts = recursively_transform(
        X,
        y,
        artifacts,
        child_transform,
        stage,
        backend,
        tqdm,
    )
    return transformations, X, y, artifacts


def __process_secondary_child_transform(
    composite: Composite,
    index: int,
    child_transform: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | None,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: list[pd.DataFrame] | None,
    tqdm: tqdm | None = None,
) -> tuple[Pipeline, X, Artifact]:
    X, y, artifacts = composite.preprocess_secondary(
        X=X,
        y=y,
        artifact=artifacts,
        results_primary=results_primary,
        index=index,
        fit=stage.is_fit(),
    )
    return recursively_transform(
        X,
        y,
        artifacts,
        child_transform,
        stage,
        backend,
        tqdm,
    )


def _backtest_on_window(
    trained_pipelines: TrainedPipelines,
    split: Fold,
    X: pd.DataFrame,
    y: pd.Series,
    artifact: Artifact,
    backend: Backend,
    mutate: bool,
) -> tuple[X, Artifact]:
    pd.options.mode.copy_on_write = True
    current_pipeline = [
        pipeline_over_time.loc[split.index] for pipeline_over_time in trained_pipelines
    ]
    if not mutate:
        current_pipeline = deepcopy_pipelines(current_pipeline)

    X_test = _cut_to_backtesting_window(X, split, current_pipeline)
    y_test = _cut_to_backtesting_window(y, split, current_pipeline)
    artifact_test = _cut_to_backtesting_window(artifact, split, current_pipeline)

    original_idx = X_test.index
    X_test, y_test, artifact_test = _get_X_y_based_on_events(
        X_test, y_test, artifact_test
    )
    artifacts_to_check = Artifact.get_events(artifact_test)
    if artifacts_to_check is None:
        artifacts_to_check = artifact_test
    assert artifacts_to_check.dropna().shape[0] == y_test.shape[0]

    results, artifacts = recursively_transform(
        X=X_test,
        y=y_test,
        artifacts=artifact_test,
        transformations=current_pipeline,
        stage=Stage.infer,
        backend=backend,
    )[1:]
    if len(results.index) != len(original_idx):
        results = results.reindex(original_idx)
        artifacts = artifacts.reindex(original_idx)
    return (
        results.loc[X.index[split.test_indices()]],
        artifacts.loc[X.index[split.test_indices()]],
    )


def _train_on_window(
    X: pd.DataFrame,
    y: pd.Series,
    artifact: Artifact,
    pipeline: Pipeline,
    split: Fold,
    backend: Backend,
    project_name: str,
    project_hyperparameters: dict | None = None,
    preprocessing_max_memory_size: int | None = None,
    show_progress: bool = False,
) -> tuple[int, TrainedPipeline, X, Artifact]:
    pd.options.mode.copy_on_write = True

    X_train: pd.DataFrame = _cut_to_train_window(X, split, Stage.inital_fit)
    y_train = _cut_to_train_window(y, split, Stage.inital_fit)

    artifact_train = _cut_to_train_window(artifact, split, Stage.inital_fit)

    original_idx = X_train.index
    X_train, y_train, artifact_train = _get_X_y_based_on_events(
        X_train, y_train, artifact_train
    )

    pipeline = deepcopy_pipelines(pipeline)
    pipeline = _set_metadata(
        pipeline,
        BlockMetadata(
            project_name=project_name,
            project_hyperparameters=project_hyperparameters,
            fold_index=split.index,
            target=y.name,
            inference=False,
            preprocessing_max_memory_size=preprocessing_max_memory_size or 0,
        ),
    )
    trained_pipeline, X_train, artifact_train = recursively_transform(
        X=X_train,
        y=y_train,
        artifacts=artifact_train,
        transformations=pipeline,
        stage=Stage.inital_fit,
        backend=backend,
        tqdm=tqdm() if show_progress else None,
    )
    if len(X_train.index) != len(original_idx):
        X_train = X_train.reindex(original_idx)
        artifact_train = artifact_train.reindex(original_idx)

    return split.index, trained_pipeline, X_train, artifact_train


def _get_X_y_based_on_events(
    X: pd.DataFrame,
    y: pd.Series,
    artifact: Artifact,
) -> tuple[pd.DataFrame, pd.Series, Artifact]:
    events = Artifact.get_events(artifact)
    if events is None:
        return X, y, artifact
    events = events.dropna()
    # if events.index.equals(X.index):
    #     return (
    #         X,
    #         events.event_label,
    #         artifact
    #         if artifact.index.equals(events.index)
    #         else artifact.loc[events.index],
    #     )
    assert len(events) > 0, "No events found in fold"
    return X.loc[events.index], events.event_label, artifact.loc[events.index]
