# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import pandas as pd

from ..base import (
    Artifact,
    Composite,
    Optimizer,
    Pipeline,
    TrainedPipelines,
    Transformation,
    Transformations,
    X,
)
from ..models.base import Model
from ..splitters import Fold
from ..utils.checks import is_prediction, is_X_available
from ..utils.dataframe import concat_on_columns, concat_on_index
from ..utils.trim import trim_initial_nans, trim_initial_nans_single
from .backend import get_backend_dependent_functions
from .memory import postprocess_X_y_into_memory_, preprocess_X_y_with_memory
from .types import Backend, Stage


def recursively_transform(
    X: X,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    artifacts: Artifact,
    transformations: Transformations,
    stage: Stage,
    backend: Backend,
) -> Tuple[X, Artifact]:
    """
    The main function to transform (and fit or update) a pipline of transformations.
    `stage` is used to determine whether to run the inner loop for online models.
    """
    if sample_weights is not None and len(X) != len(sample_weights):
        sample_weights = sample_weights.loc[
            X.index
        ]  # we're calling recursively_transform() recursively, and we trim sample_weights as well, but for simplicity's sake, recursive_transform doesn't return it explicitly, so these two could get out of sync.

    if y is not None and len(X) != len(y):
        y = y[X.index]

    if isinstance(transformations, List):
        for transformation in transformations:
            X, artifacts = recursively_transform(
                X, y, sample_weights, artifacts, transformation, stage, backend
            )
        return X, artifacts

    elif isinstance(transformations, Composite):
        return _process_composite(
            transformations, X, y, sample_weights, artifacts, stage, backend
        )

    elif isinstance(transformations, Optimizer):
        return _process_optimizer(
            transformations, X, y, sample_weights, artifacts, stage, backend
        )

    elif isinstance(transformations, Transformation) or isinstance(
        transformations, Model
    ):
        # If the transformation needs to be "online", and we're in the update stage, we need to run the inner loop.
        if (
            transformations.properties.mode == Transformation.Properties.Mode.online
            and stage in [Stage.update, Stage.update_online_only]
            and not transformations.properties._internal_supports_minibatch_backtesting
        ):
            return _process_with_inner_loop(
                transformations, X, y, sample_weights, artifacts
            )
        # If the transformation is "online" but also supports our internal "mini-batch"-style updating
        elif (
            transformations.properties.mode == Transformation.Properties.Mode.online
            and stage in [Stage.update, Stage.update_online_only]
            and transformations.properties._internal_supports_minibatch_backtesting
        ):
            return _process_internal_online_model_minibatch_inference_and_update(
                transformations, X, y, sample_weights, artifacts
            )

        # or perform "mini-batch" updating OR the initial fit.
        else:
            return _process_minibatch_transformation(
                transformations, X, y, sample_weights, artifacts, stage
            )

    else:
        raise ValueError(
            f"{transformations} is not a Fold Transformation, but of type"
            f" {type(transformations)}"
        )


def _process_composite(
    composite: Composite,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
) -> Tuple[X, Artifact]:
    backend_functions = get_backend_dependent_functions(backend)

    composite.before_fit(X)
    primary_transformations = composite.get_child_transformations_primary()

    (results_primary, artifacts_primary,) = zip(
        *backend_functions.process_child_transformations(
            __process_primary_child_transform,
            enumerate(primary_transformations),
            composite,
            X,
            y,
            sample_weights,
            artifacts,
            stage,
            backend,
            None,
        )
    )

    if composite.properties.primary_only_single_pipeline:
        assert len(results_primary) == 1, ValueError(
            "Expected single output from primary transformations, got"
            f" {len(results_primary)} instead."
        )
    if composite.properties.primary_requires_predictions:
        assert is_prediction(results_primary[0]), ValueError(
            "Expected predictions from primary transformations, but got something else."
        )

    secondary_transformations = composite.get_child_transformations_secondary()

    artifacts_primary = composite.postprocess_artifacts_primary(artifacts_primary)
    if secondary_transformations is None:
        return (
            composite.postprocess_result_primary(results_primary, y),
            artifacts_primary,
        )

    (results_secondary, artifacts_secondary,) = zip(
        *backend_functions.process_child_transformations(
            __process_secondary_child_transform,
            enumerate(secondary_transformations),
            composite,
            X,
            y,
            sample_weights,
            artifacts,
            stage,
            backend,
            results_primary,
        )
    )

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
        composite.postprocess_result_secondary(
            results_primary, results_secondary, y, in_sample=stage == Stage.inital_fit
        ),
        composite.postprocess_artifacts_secondary(
            artifacts_primary, artifacts_secondary
        ),
    )


def _process_optimizer(
    optimizer: Optimizer,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
) -> Tuple[X, Artifact]:
    backend_functions = get_backend_dependent_functions(backend)

    optimized_pipeline = optimizer.get_optimized_pipeline()
    artifact = None
    if optimized_pipeline is None:
        candidates = optimizer.get_candidates()

        results_primary, _ = zip(
            *backend_functions.process_child_transformations(
                __process_candidates,
                enumerate(candidates),
                optimizer,
                X,
                y,
                sample_weights,
                artifacts,
                stage,
                backend,
                None,
            )
        )
        results_primary = [
            trim_initial_nans_single(result) for result in results_primary
        ]
        artifact = optimizer.process_candidate_results(
            results_primary, y.loc[results_primary[0].index]
        )

    optimized_pipeline = optimizer.get_optimized_pipeline()
    return recursively_transform(
        X,
        y,
        sample_weights,
        concat_on_columns([artifact, artifacts]),
        optimized_pipeline,
        stage,
        backend,
    )


def _process_with_inner_loop(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    artifacts: Artifact,
) -> Tuple[X, Artifact]:
    if len(X) == 0:
        return (pd.DataFrame(), pd.DataFrame())

    # We need to run the inference & fit loop on each row, sequentially (one-by-one).
    # This is so the transformation can update its parameters after each sample.

    def transform_row(
        X_row: pd.DataFrame,
        y_row: Optional[pd.Series],
        sample_weights_row: Optional[pd.Series],
    ):
        (
            X_row_with_memory,
            y_row_with_memory,
            sample_weights_row_with_memory,
        ) = preprocess_X_y_with_memory(
            transformation, X_row, y_row, sample_weights_row, in_sample=False
        )
        result, _ = transformation.transform(X_row_with_memory, in_sample=False)
        if y_row is not None:
            artifact = transformation.update(
                X_row_with_memory, y_row_with_memory, sample_weights_row_with_memory
            )
            _ = concat_on_columns([artifact, artifacts])
            postprocess_X_y_into_memory_(
                transformation,
                X_row_with_memory,
                y_row_with_memory,
                sample_weights_row_with_memory,
                False,
            )
        return result.loc[X_row.index]

    return (
        concat_on_index(
            [
                transform_row(
                    X.loc[index:index],
                    y.loc[index:index] if y is not None else None,
                    sample_weights.loc[index] if sample_weights is not None else None,
                )
                for index in X.index
            ]
        ),
        pd.DataFrame(),
    )


def _process_internal_online_model_minibatch_inference_and_update(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    artifacts: Artifact,
) -> Tuple[X, Artifact]:
    X, y, sample_weights = trim_initial_nans(X, y, sample_weights)
    (
        X_with_memory,
        y_with_memory,
        sample_weights_with_memory,
    ) = preprocess_X_y_with_memory(transformation, X, y, sample_weights, in_sample=True)
    postprocess_X_y_into_memory_(
        transformation, X_with_memory, y_with_memory, sample_weights_with_memory, True
    )
    return_value, artifact = transformation.transform(X_with_memory, in_sample=True)
    artifacts = concat_on_columns([artifact, artifacts])

    artifact = transformation.update(X_with_memory, y_with_memory, sample_weights)
    postprocess_X_y_into_memory_(transformation, X, y, sample_weights, False)
    return return_value.loc[X.index], concat_on_columns([artifact, artifacts])


def _process_minibatch_transformation(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    artifacts: Artifact,
    stage: Stage,
) -> Tuple[X, Artifact]:
    X, y, sample_weights = trim_initial_nans(X, y, sample_weights)

    if not is_X_available(X) and transformation.properties.requires_X:
        raise ValueError(
            "X is None, but transformation"
            f" {transformation.__class__.__name__} requires it."
        )

    in_sample = stage == Stage.inital_fit
    (
        X_with_memory,
        y_with_memory,
        sample_weights_with_memory,
    ) = preprocess_X_y_with_memory(
        transformation, X, y, sample_weights, in_sample=in_sample
    )
    # The order is:
    # 1. fit (if we're in the initial_fit stage)
    artifact = None
    if stage == Stage.inital_fit:
        artifact = transformation.fit(
            X_with_memory, y_with_memory, sample_weights_with_memory
        )
        postprocess_X_y_into_memory_(
            transformation,
            X_with_memory,
            y_with_memory,
            sample_weights_with_memory,
            in_sample=stage == Stage.inital_fit,
        )
        artifacts = concat_on_columns([artifact, artifacts])
    # 2. transform (inference)
    (
        X_with_memory,
        y_with_memory,
        sample_weights_with_memory,
    ) = preprocess_X_y_with_memory(
        transformation, X, y, sample_weights, in_sample=False
    )
    return_value, artifact = transformation.transform(
        X_with_memory, in_sample=in_sample
    )
    artifacts = concat_on_columns([artifact, artifacts])
    # 3. update (if we're in the update stage)
    if stage == Stage.update:
        artifact = transformation.update(X_with_memory, y_with_memory, sample_weights)
        artifacts = concat_on_columns([artifact, artifacts])
        postprocess_X_y_into_memory_(transformation, X, y, sample_weights, False)
    return return_value.loc[X.index], artifacts


def __process_candidates(
    optimizer: Optimizer,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: Optional[List[pd.DataFrame]],
) -> Tuple[X, Artifact]:
    splits = optimizer.splitter.splits(length=len(y))

    (
        processed_idx,
        processed_pipelines,
        processed_artifacts,
    ) = _sequential_train_on_window(
        child_transform, X, y, splits, sample_weights, backend
    )
    trained_pipelines = _extract_trained_pipelines(processed_idx, processed_pipelines)

    result = _backtest_on_window(
        trained_pipelines,
        splits[0],
        X,
        y,
        sample_weights,
        backend,
        mutate=False,
    )
    return (
        trim_initial_nans_single(result),
        processed_artifacts[0],
    )


def __process_primary_child_transform(
    composite: Composite,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: Optional[List[pd.DataFrame]],
) -> Tuple[X, Artifact]:
    X, y, sample_weights = composite.preprocess_primary(
        X, index, y, sample_weights=sample_weights, fit=stage.is_fit_or_update()
    )
    return recursively_transform(
        X, y, sample_weights, artifacts, child_transform, stage, backend
    )


def __process_secondary_child_transform(
    composite: Composite,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: Optional[List[pd.DataFrame]],
) -> Tuple[X, Artifact]:
    X, y = composite.preprocess_secondary(
        X, y, results_primary, index, fit=stage.is_fit_or_update()
    )
    return recursively_transform(
        X, y, sample_weights, artifacts, child_transform, stage, backend
    )


def deepcopy_pipelines(transformation: Transformations) -> Transformations:
    if isinstance(transformation, List):
        return [deepcopy_pipelines(t) for t in transformation]
    elif isinstance(transformation, Composite):
        return transformation.clone(deepcopy_pipelines)
    else:
        return deepcopy(transformation)


def _train_on_window(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series],
    pipeline: Pipeline,
    split: Fold,
    never_update: bool,
    backend: Backend,
) -> Tuple[int, Pipeline, Artifact]:
    stage = Stage.inital_fit if (split.order == 0 or never_update) else Stage.update
    window_start = (
        split.update_window_start if stage == Stage.update else split.train_window_start
    )
    window_end = (
        split.update_window_end if stage == Stage.update else split.train_window_end
    )
    X_train: pd.DataFrame = X.iloc[window_start:window_end]  # type: ignore
    y_train = y.iloc[window_start:window_end]

    sample_weights_train = (
        sample_weights.iloc[window_start:window_end]
        if sample_weights is not None
        else None
    )
    artifacts = pd.DataFrame()

    pipeline = deepcopy_pipelines(pipeline)
    X_train, artifacts = recursively_transform(
        X_train, y_train, sample_weights_train, artifacts, pipeline, stage, backend
    )

    return split.model_index, pipeline, artifacts


def _backtest_on_window(
    trained_pipelines: TrainedPipelines,
    split: Fold,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series],
    backend: Backend,
    mutate: bool,
) -> pd.DataFrame:
    current_pipeline = [
        pipeline_over_time.loc[split.model_index]
        for pipeline_over_time in trained_pipelines
    ]
    if not mutate:
        current_pipeline = deepcopy_pipelines(current_pipeline)

    X_test = X.iloc[split.test_window_start : split.test_window_end]
    y_test = y.iloc[split.test_window_start : split.test_window_end]
    sample_weights_test = (
        sample_weights.iloc[split.train_window_start : split.test_window_end]
        if sample_weights is not None
        else None
    )
    return recursively_transform(
        X_test,
        y_test,
        sample_weights_test,
        pd.DataFrame(),
        current_pipeline,
        stage=Stage.update_online_only,
        backend=backend,
    )[0]


def _sequential_train_on_window(
    pipeline: Pipeline,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splits: List[Fold],
    sample_weights: Optional[pd.Series] = None,
    backend: Union[Backend, str] = Backend.no,
) -> Tuple[List[int], List[Pipeline], List[Artifact]]:
    processed_idx = []
    processed_pipelines: List[Pipeline] = []
    processed_pipeline = pipeline
    processed_artifacts = []
    for split in splits:
        processed_id, processed_pipeline, processed_artifact = _train_on_window(
            X,
            y,
            sample_weights,
            processed_pipeline,
            split,
            False,
            backend,
        )
        processed_idx.append(processed_id)
        processed_pipelines.append(processed_pipeline)
        processed_artifacts.append(processed_artifact)

    return processed_idx, processed_pipelines, processed_artifacts


def _extract_trained_pipelines(
    processed_idx: List[int], processed_pipelines: List[Pipeline]
) -> TrainedPipelines:
    return [
        pd.Series(
            transformation_over_time,
            index=processed_idx,
            name=transformation_over_time[0].name,
        )
        for transformation_over_time in zip(*processed_pipelines)
    ]
