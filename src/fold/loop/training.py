# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Optional, Tuple, Union

import pandas as pd

from ..base import (
    Artifact,
    DeployablePipeline,
    EventDataFrame,
    InSamplePredictions,
    Pipeline,
    PipelineCard,
    TrainedPipelineCard,
)
from ..splitters import Fold, SlidingWindowSplitter, Splitter
from ..utils.dataframe import concat_on_index_override_duplicate_rows
from ..utils.list import (
    unpack_list_of_tuples,
    wrap_in_double_list_if_needed,
    wrap_in_list,
)
from ..utils.trim import trim_initial_nans
from .backend import get_backend
from .checks import check_types
from .common import _sequential_train_on_window, _train_on_window
from .types import Backend, BackendType, TrainMethod
from .utils import _extract_trained_pipelines
from .wrap import wrap_transformation_if_needed


def train(
    pipelinecard: Union[PipelineCard, Pipeline],
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    sample_weights: Optional[pd.Series] = None,
    events: Optional[EventDataFrame] = None,
    train_method: Union[TrainMethod, str] = TrainMethod.parallel,
    backend: Union[BackendType, Backend, str] = BackendType.no,
    silent: bool = False,
    return_artifacts: bool = False,
    return_insample: bool = False,
) -> Union[
    TrainedPipelineCard,
    Tuple[TrainedPipelineCard, Artifact],
    Tuple[TrainedPipelineCard, Artifact, InSamplePredictions],
]:
    """
    Trains a pipeline on a given dataset, for all folds returned by the Splitter.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline to be fitted.
    X: Optional[pd.DataFrame]
        Exogenous Data.
    y: pd.Series
        Endogenous Data (Target).
    splitter: Splitter
        Defines how the folds should be constructed.
    train_method : TrainMethod, str = TrainMethod.parallel
        The training methodology, by default `parallel`.
    backend: str, BackendType = BackendType.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    sample_weights: Optional[pd.Series] = None
        Weights assigned to each sample/timestamp, that are passed into models that support it, by default None.
    events: EventDataFrame, optional = None
        Events that should be passed into the pipeline, by default None.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    return_artifacts: bool = False
        Whether to return the artifacts of the training process, by default False.
    return_insample: bool = False

    Returns
    -------
    TrainedPipelineCard
        The fitted pipelines, for all folds.
    """
    pipelinecard = (
        pipelinecard
        if isinstance(pipelinecard, PipelineCard)
        else PipelineCard(preprocessing=None, pipeline=pipelinecard)
    )
    X, y = check_types(X, y)
    artifact = Artifact.from_events_sample_weights(X.index, events, sample_weights)
    X, y, artifact = trim_initial_nans(X, y, artifact)
    train_method = TrainMethod.from_str(train_method)
    backend = get_backend(backend)

    if pipelinecard.preprocessing is not None:
        preprocessing_pipeline = wrap_in_double_list_if_needed(
            pipelinecard.preprocessing
        )
        preprocessing_pipeline = wrap_transformation_if_needed(preprocessing_pipeline)
        (
            _,
            trained_preprocessing_pipeline,
            preprocessed_X,
            preprocessed_artifact,
        ) = _train_on_window(
            X,
            y,
            artifact,
            preprocessing_pipeline,
            Fold(0, 0, 0, len(X), 0, 0, 0, len(X)),
            never_update=True,
            backend=backend,
        )
        assert preprocessed_X.shape[0] == X.shape[0]
        assert preprocessed_artifact.shape[0] == artifact.shape[0]
        X = preprocessed_X
        artifact = preprocessed_artifact

    if isinstance(splitter, SlidingWindowSplitter):
        assert train_method != TrainMethod.sequential, (
            "SlidingWindowSplitter is conceptually incompatible with"
            " TrainMethod.sequential"
        )

    pipeline = wrap_in_list(pipelinecard.pipeline)
    pipeline = wrap_transformation_if_needed(pipeline)

    splits = splitter.splits(index=X.index)
    if len(splits) == 0:
        raise ValueError("No splits were generated by the Splitter.")

    if train_method == TrainMethod.parallel_with_search and len(splits) > 1:
        (
            first_batch_index,
            first_batch_transformations,
            first_batch_predictions,
            first_batch_artifacts,
        ) = _train_on_window(
            X,
            y,
            artifact,
            pipeline,
            splits[0],
            never_update=True,
            backend=backend,
        )

        (
            rest_idx,
            rest_transformations,
            rest_predictions,
            rest_artifacts,
        ) = unpack_list_of_tuples(
            backend.train_pipeline(
                _train_on_window,
                first_batch_transformations,
                X,
                y,
                artifact,
                splits[1:],
                False,
                backend,
                silent,
            )
        )
        processed_idx = [first_batch_index] + list(rest_idx)
        processed_pipelines = [first_batch_transformations] + list(rest_transformations)
        processed_predictions = [first_batch_predictions] + list(rest_predictions)
        processed_artifacts = [first_batch_artifacts] + list(rest_artifacts)

    elif train_method == TrainMethod.parallel:
        (
            processed_idx,
            processed_pipelines,
            processed_predictions,
            processed_artifacts,
        ) = unpack_list_of_tuples(
            backend.train_pipeline(
                _train_on_window,
                pipeline,
                X,
                y,
                artifact,
                splits,
                True,
                backend,
                silent,
            )
        )

    else:
        (
            processed_idx,
            processed_pipelines,
            processed_predictions,
            processed_artifacts,
        ) = _sequential_train_on_window(pipeline, X, y, splits, artifact, backend)

    trained_pipelines = TrainedPipelineCard(
        preprocessing=_extract_trained_pipelines(
            [0] * len(trained_preprocessing_pipeline), trained_preprocessing_pipeline
        )
        if pipelinecard.preprocessing
        else None,
        pipeline=_extract_trained_pipelines(processed_idx, processed_pipelines),
    )

    if return_artifacts is True:
        processed_artifacts = concat_on_index_override_duplicate_rows(
            processed_artifacts
        )
        processed_predictions = concat_on_index_override_duplicate_rows(
            processed_predictions
        )
        assert processed_artifacts.index.equals(processed_predictions.index)
        if return_insample:
            return (
                trained_pipelines,
                processed_artifacts,
                processed_predictions,
            )
        else:
            return trained_pipelines, processed_artifacts
    else:
        if return_insample:
            return trained_pipelines, concat_on_index_override_duplicate_rows(
                processed_predictions
            )
        else:
            return trained_pipelines


def train_for_deployment(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series] = None,
    events: Optional[EventDataFrame] = None,
) -> DeployablePipeline:
    X, y = check_types(X, y)
    artifact = Artifact.from_events_sample_weights(X.index, events, sample_weights)
    X, y, artifact = trim_initial_nans(X, y, artifact)

    pipeline = wrap_in_list(pipeline)
    pipeline = wrap_transformation_if_needed(pipeline)
    _, transformations, predictions, artifacts = _train_on_window(
        X,
        y,
        artifact,
        pipeline,
        Fold(
            order=0,
            model_index=0,
            train_window_start=0,
            train_window_end=None,
            update_window_start=0,
            update_window_end=0,
            test_window_start=0,
            test_window_end=None,
        ),
        never_update=True,
        backend=get_backend(BackendType.no),
    )
    return transformations
