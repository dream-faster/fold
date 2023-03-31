from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Tuple

import pandas as pd

from ..composites.base import Composite
from ..models.base import Model
from ..transformations.base import Transformation, Transformations
from ..utils.checks import is_prediction, is_X_available
from ..utils.pandas import trim_initial_nans
from .backend import get_backend_dependent_functions
from .types import Backend, Stage


def recursively_transform(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    transformations: Transformations,
    stage: Stage,
    backend: Backend,
) -> pd.DataFrame:
    """
    The main function to transform (and fit or update) a pipline of transformations.
    `stage` is used to determine whether to run the inner loop for online models.
    """
    if y is not None and len(X) != len(y):
        y = y[X.index]

    if isinstance(transformations, List):
        for transformation in transformations:
            X = recursively_transform(
                X, y, sample_weights, transformation, stage, backend
            )
        return X

    elif isinstance(transformations, Composite):
        return process_composite(transformations, X, y, sample_weights, stage, backend)

    elif isinstance(transformations, Transformation) or isinstance(
        transformations, Model
    ):
        # If the transformation needs to be "online", and we're in the update stage, we need to run the inner loop.
        if (
            transformations.properties.mode == Transformation.Properties.Mode.online
            and stage in [Stage.update, Stage.update_online_only]
            and not transformations.properties._internal_supports_minibatch_backtesting
        ):
            return process_with_inner_loop(transformations, X, y, sample_weights)
        # If the transformation is "online" but also supports our internal "mini-batch"-style updating
        elif (
            transformations.properties.mode == Transformation.Properties.Mode.online
            and stage in [Stage.update, Stage.update_online_only]
            and transformations.properties._internal_supports_minibatch_backtesting
        ):
            return process_internal_online_model_minibatch_inference_and_update(
                transformations, X, y, sample_weights
            )

        # or perform "mini-batch" updating OR the initial fit.
        else:
            return process_minibatch_transformation(
                transformations, X, y, sample_weights, stage
            )

    else:
        raise ValueError(
            f"{transformations} is not a Fold Transformation, but of type"
            f" {type(transformations)}"
        )


def process_composite(
    composite: Composite,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    stage: Stage,
    backend: Backend,
) -> pd.DataFrame:
    backend_functions = get_backend_dependent_functions(backend)

    composite.before_fit(X)
    primary_transformations = composite.get_child_transformations_primary()

    results_primary = backend_functions.process_primary_child_transformations(
        __process_primary_child_transform,
        enumerate(primary_transformations),
        composite,
        X,
        y,
        sample_weights,
        stage,
        backend,
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

    if secondary_transformations is None:
        return composite.postprocess_result_primary(results_primary, y)

    results_secondary = backend_functions.process_secondary_child_transformations(
        __process_secondary_child_transform,
        enumerate(secondary_transformations),
        composite,
        X,
        y,
        sample_weights,
        results_primary,
        stage,
        backend,
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

    return composite.postprocess_result_secondary(results_primary, results_secondary, y)


def process_with_inner_loop(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
) -> pd.DataFrame:
    if len(X) == 0:
        return pd.DataFrame()

    # We need to run the inference & fit loop on each row, sequentially (one-by-one).
    # This is so the transformation can update its parameters after each sample.

    def transform_row(
        X_row: pd.DataFrame, y_row: Optional[pd.Series], sample_weights_row
    ):
        X_row_with_memory, y_row_with_memory = __preprocess_X_y_with_memory(
            transformation, X_row, y_row
        )
        result = transformation.transform(X_row_with_memory, in_sample=False)
        if y_row is not None:
            transformation.update(
                X_row_with_memory, y_row_with_memory, sample_weights_row
            )
            __postprocess_X_y_into_memory(
                transformation, X_row_with_memory, y_row_with_memory, False
            )
        return result.loc[X_row.index]

    return pd.concat(
        [
            transform_row(
                X.loc[index:index],
                y.loc[index:index] if y is not None else None,
                sample_weights.loc[index] if sample_weights is not None else None,
            )
            for index in X.index
        ],
        axis="index",
    )


def process_internal_online_model_minibatch_inference_and_update(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
) -> pd.DataFrame:
    X, y = trim_initial_nans(X, y)
    X_with_memory, y_with_memory = __preprocess_X_y_with_memory(transformation, X, y)
    __postprocess_X_y_into_memory(transformation, X_with_memory, y_with_memory, True)
    return_value = transformation.transform(X_with_memory, in_sample=True)

    transformation.update(X_with_memory, y_with_memory, sample_weights)
    __postprocess_X_y_into_memory(transformation, X, y, False)
    return return_value.loc[X.index]


def process_minibatch_transformation(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    stage: Stage,
) -> pd.DataFrame:
    X, y = trim_initial_nans(X, y)

    if not is_X_available(X) and transformation.properties.requires_X:
        raise ValueError(
            f"X is None, but transformation {transformation.__class__.__name__} requires it."
        )

    X_with_memory, y_with_memory = __preprocess_X_y_with_memory(transformation, X, y)
    # The order is:
    # 1. fit (if we're in the initial_fit stage)
    if stage == Stage.inital_fit:
        transformation.fit(X_with_memory, y_with_memory, sample_weights)
        __postprocess_X_y_into_memory(
            transformation,
            X_with_memory,
            y_with_memory,
            in_sample=stage == Stage.inital_fit,
        )
    # 2. transform (inference)
    X_with_memory, y_with_memory = __preprocess_X_y_with_memory(transformation, X, y)
    return_value = transformation.transform(
        X_with_memory, in_sample=stage == Stage.inital_fit
    )
    # 3. update (if we're in the update stage)
    if stage == Stage.update:
        transformation.update(X_with_memory, y_with_memory, sample_weights)
        __postprocess_X_y_into_memory(transformation, X, y, False)
    return return_value.loc[X.index]


def __process_primary_child_transform(
    composite: Composite,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    stage: Stage,
    backend: Backend,
) -> pd.DataFrame:
    X, y = composite.preprocess_primary(X, index, y, fit=stage.is_fit_or_update())
    return recursively_transform(X, y, sample_weights, child_transform, stage, backend)


def __process_secondary_child_transform(
    composite: Composite,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    results_primary: List[pd.DataFrame],
    stage: Stage,
    backend: Backend,
) -> pd.DataFrame:
    X, y = composite.preprocess_secondary(
        X, y, results_primary, index, fit=stage.is_fit_or_update()
    )
    return recursively_transform(X, y, sample_weights, child_transform, stage, backend)


def deepcopy_pipelines(transformation: Transformations) -> Transformations:
    if isinstance(transformation, List):
        return [deepcopy_pipelines(t) for t in transformation]
    elif isinstance(transformation, Composite):
        return transformation.clone(deepcopy_pipelines)
    else:
        return deepcopy(transformation)


def __preprocess_X_y_with_memory(
    transformation: Transformation, X: pd.DataFrame, y: Optional[pd.Series]
) -> Tuple[pd.DataFrame, pd.Series]:
    if transformation._state is None or transformation.properties.memory_size is None:
        return X, y
    memory_X, memory_y = transformation._state.memory_X, transformation._state.memory_y
    non_overlapping_indices = ~memory_X.index.isin(X.index)
    memory_X, memory_y = (
        memory_X[non_overlapping_indices],
        memory_y[non_overlapping_indices],
    )
    if len(memory_X) == 0:
        return X, y
    assert len(memory_X) == len(memory_y)
    if y is None:
        return pd.concat([memory_X, X], axis="index"), y
    else:
        memory_y.name = y.name
        return pd.concat([memory_X, X], axis="index"), pd.concat(
            [memory_y, y], axis="index"
        )


def __postprocess_X_y_into_memory(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    in_sample: bool,
) -> None:
    # don't update the transformation if we're in inference mode (y is None)
    if transformation.properties.memory_size is None or y is None:
        return

    window_size = (
        len(X)
        if transformation.properties.memory_size == 0
        else transformation.properties.memory_size
    )
    if in_sample:
        # store the whole training X and y
        transformation._state = Transformation.State(
            memory_X=X,
            memory_y=y,
        )
    elif transformation.properties.memory_size < len(X):
        memory_X, memory_y = (
            transformation._state.memory_X,
            transformation._state.memory_y,
        )
        memory_y.name = y.name
        #  memory requirement is greater than the current batch, so we use the previous memory as well
        transformation._state = Transformation.State(
            memory_X=pd.concat([memory_X, X], axis="index").iloc[-window_size:],
            memory_y=pd.concat([memory_y, y], axis="index").iloc[-window_size:],
        )
    else:
        transformation._state = Transformation.State(
            memory_X=X.iloc[-window_size:],
            memory_y=y.iloc[-window_size:],
        )
