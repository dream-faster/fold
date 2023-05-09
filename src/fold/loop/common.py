# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Tuple

import pandas as pd

from fold.splitters import Fold, SingleWindowSplitter

from ..base import (
    Artifact,
    Composite,
    Optimizer,
    Pipeline,
    Transformation,
    Transformations,
    X,
)
from ..models.base import Model
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
        # Optimized needs to run the search
        candidates = optimizer.get_candidates()

        splitter = SingleWindowSplitter(0.8)
        splits = splitter.splits(length=len(y))

        processed_idx = []
        processed_pipelines = []
        processed_pipeline = None
        processed_artifacts = []
        for split in splits:
            (
                processed_id,
                processed_pipeline,
                processed_artifact,
            ) = _process_pipeline_window(
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

        _, _ = zip(
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
        artifact = optimizer.process_candidate_results(results_primary, y)

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
    return recursively_transform(
        X, y, sample_weights, artifacts, child_transform, stage, backend
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
    X, y = composite.preprocess_primary(X, index, y, fit=stage.is_fit_or_update())
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


def _process_pipeline_window(
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
