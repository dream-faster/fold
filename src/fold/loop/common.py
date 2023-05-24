# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import List, Optional, Tuple, TypeVar, Union

import pandas as pd

from ..base import (
    Artifact,
    Composite,
    Extras,
    Optimizer,
    Pipeline,
    TrainedPipeline,
    TrainedPipelines,
    Transformation,
    Transformations,
    X,
)
from ..models.base import Model
from ..splitters import Fold
from ..utils.checks import is_prediction
from ..utils.dataframe import concat_on_columns
from ..utils.trim import trim_initial_nans_single
from .backend import get_backend_dependent_functions
from .process.process_inner_loop import _process_with_inner_loop
from .process.process_minibatch import (
    _process_internal_online_model_minibatch_inference_and_update,
    _process_minibatch_transformation,
)
from .types import Backend, Stage
from .utils import _extract_trained_pipelines, deepcopy_pipelines, replace_with

T = TypeVar(
    "T",
    bound=Union[
        Transformation,
        Composite,
        Optimizer,
        List[Union[Transformation, Optimizer, Composite]],
    ],
)


def recursively_transform(
    X: X,
    y: Optional[pd.Series],
    extras: Extras,
    artifacts: Artifact,
    transformations: T,
    stage: Stage,
    backend: Backend,
) -> Tuple[T, X, Artifact]:
    """
    The main function to transform (and fit or update) a pipline of transformations.
    `stage` is used to determine whether to run the inner loop for online models.
    """
    if len(X) != len(extras):
        extras = extras.loc(
            X.index
        )  # we're calling recursively_transform() recursively, and we trim extras as well, but for simplicity's sake, recursive_transform doesn't return it explicitly, so these two could get out of sync.

    if y is not None and len(X) != len(y):
        y = y[X.index]

    if isinstance(transformations, List):
        processed_transformations = []
        for transformation in transformations:
            processed_transformation, X, artifacts = recursively_transform(
                X, y, extras, artifacts, transformation, stage, backend
            )
            processed_transformations.append(processed_transformation)
        return processed_transformations, X, artifacts

    elif isinstance(transformations, Composite):
        return _process_composite(
            transformations, X, y, extras, artifacts, stage, backend
        )

    elif isinstance(transformations, Optimizer):
        return _process_optimizer(
            transformations, X, y, extras, artifacts, stage, backend
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
            return _process_with_inner_loop(transformations, X, y, extras, artifacts)
        # If the transformation is "online" but also supports our internal "mini-batch"-style updating
        elif (
            transformations.properties.mode == Transformation.Properties.Mode.online
            and stage in [Stage.update, Stage.update_online_only]
            and transformations.properties._internal_supports_minibatch_backtesting
        ):
            return _process_internal_online_model_minibatch_inference_and_update(
                transformations, X, y, extras, artifacts
            )

        # or perform "mini-batch" updating OR the initial fit.
        else:
            return _process_minibatch_transformation(
                transformations, X, y, extras, artifacts, stage
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
    extras: Extras,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
) -> Tuple[Composite, X, Artifact]:
    backend_functions = get_backend_dependent_functions(backend)

    composite.before_fit(X)
    primary_transformations = composite.get_children_primary()

    primary_transformations, results_primary, artifacts_primary = zip(
        *backend_functions.process_child_transformations(
            __process_primary_child_transform,
            enumerate(primary_transformations),
            composite,
            X,
            y,
            extras,
            artifacts,
            stage,
            backend,
            None,
        )
    )
    composite = composite.clone(replace_with(primary_transformations[0]))

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

    artifacts_primary = composite.postprocess_artifacts_primary(
        artifacts_primary, extras=extras
    )
    if secondary_transformations is None:
        return (
            composite,
            composite.postprocess_result_primary(results_primary, y),
            artifacts_primary,
        )

    (secondary_transformations, results_secondary, artifacts_secondary) = zip(
        *backend_functions.process_child_transformations(
            __process_secondary_child_transform,
            enumerate(secondary_transformations),
            composite,
            X,
            y,
            extras,
            artifacts,
            stage,
            backend,
            results_primary,
        )
    )
    composite = composite.clone(replace_with(secondary_transformations[0]))

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
    extras: Extras,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
) -> Tuple[Pipeline, X, Artifact]:
    backend_functions = get_backend_dependent_functions(backend)

    optimized_pipeline = optimizer.get_optimized_pipeline()
    artifact = None
    if optimized_pipeline is None:
        for candidates in optimizer.get_candidates():
            if len(candidates) == 0:
                break

            _, results, _ = zip(
                *backend_functions.process_child_transformations(
                    __process_candidates,
                    enumerate(candidates),
                    optimizer,
                    X,
                    y,
                    extras,
                    artifacts,
                    stage,
                    backend,
                    None,
                )
            )
            results = [trim_initial_nans_single(result) for result in results]
            artifact = optimizer.process_candidate_results(
                results, y.loc[results[0].index]
            )

    optimized_pipeline = optimizer.get_optimized_pipeline()[0]
    processed_optimized_pipeline, X, artifact = recursively_transform(
        X,
        y,
        extras,
        concat_on_columns([artifact, artifacts]),
        optimized_pipeline,
        stage,
        backend,
    )
    return optimizer, X, artifact


def __process_candidates(
    optimizer: Optimizer,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    extras: Extras,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: Optional[List[pd.DataFrame]],
) -> Tuple[Transformations, X, Artifact]:
    splits = optimizer.splitter.splits(length=len(y))

    (
        processed_idx,
        processed_pipelines,
        processed_artifacts,
    ) = _sequential_train_on_window(child_transform, X, y, splits, extras, backend)
    trained_pipelines = _extract_trained_pipelines(processed_idx, processed_pipelines)

    result = _backtest_on_window(
        trained_pipelines,
        splits[0],
        X,
        y,
        extras,
        backend,
        mutate=False,
    )[0]
    return (
        trained_pipelines,
        trim_initial_nans_single(result),
        processed_artifacts[0],
    )


def __process_primary_child_transform(
    composite: Composite,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    extras: Extras,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: Optional[List[pd.DataFrame]],
) -> Tuple[Transformations, X, Artifact]:
    X, y, extras = composite.preprocess_primary(
        X, index, y, extras=extras, fit=stage.is_fit_or_update()
    )
    return recursively_transform(
        X, y, extras, artifacts, child_transform, stage, backend
    )


def __process_secondary_child_transform(
    composite: Composite,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    extras: Extras,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: Optional[List[pd.DataFrame]],
) -> Tuple[Transformations, X, Artifact]:
    X, y = composite.preprocess_secondary(
        X, y, results_primary, index, fit=stage.is_fit_or_update()
    )
    return recursively_transform(
        X, y, extras, artifacts, child_transform, stage, backend
    )


def _backtest_on_window(
    trained_pipelines: TrainedPipelines,
    split: Fold,
    X: pd.DataFrame,
    y: pd.Series,
    extras: Extras,
    backend: Backend,
    mutate: bool,
) -> Tuple[X, Artifact]:
    current_pipeline = [
        pipeline_over_time.loc[split.model_index]
        for pipeline_over_time in trained_pipelines
    ]
    if not mutate:
        current_pipeline = deepcopy_pipelines(current_pipeline)

    X_test = X.iloc[split.test_window_start : split.test_window_end]
    y_test = y.iloc[split.test_window_start : split.test_window_end]
    extras_test = extras.iloc(slice(split.train_window_start, split.test_window_end))
    results, artifacts = recursively_transform(
        X_test,
        y_test,
        extras_test,
        pd.DataFrame(),
        current_pipeline,
        stage=Stage.update_online_only,
        backend=backend,
    )[1:]
    return results, artifacts.loc[X.index[split.test_window_start - 1] :]


def _train_on_window(
    X: pd.DataFrame,
    y: pd.Series,
    extras: Extras,
    pipeline: Pipeline,
    split: Fold,
    never_update: bool,
    backend: Backend,
) -> Tuple[int, TrainedPipeline, Artifact]:
    stage = Stage.inital_fit if (split.order == 0 or never_update) else Stage.update
    window_start = (
        split.update_window_start if stage == Stage.update else split.train_window_start
    )
    window_end = (
        split.update_window_end if stage == Stage.update else split.train_window_end
    )
    X_train: pd.DataFrame = X.iloc[window_start:window_end]  # type: ignore
    y_train = y.iloc[window_start:window_end]

    extras_train = extras.iloc(slice(window_start, window_end))
    artifacts = pd.DataFrame()

    pipeline = deepcopy_pipelines(pipeline)
    trained_pipeline, X_train, artifacts = recursively_transform(
        X_train, y_train, extras_train, artifacts, pipeline, stage, backend
    )

    return split.model_index, trained_pipeline, artifacts


def _sequential_train_on_window(
    pipeline: Pipeline,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splits: List[Fold],
    extras: Extras,
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
            extras,
            processed_pipeline,
            split,
            False,
            backend,
        )
        processed_idx.append(processed_id)
        processed_pipelines.append(processed_pipeline)
        processed_artifacts.append(processed_artifact)

    return processed_idx, processed_pipelines, processed_artifacts
