# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


import logging

import pandas as pd
from finml_utils.dataframes import concat_on_index_without_duplicates

from fold.base.utils import get_maximum_memory_size

from ..base import (
    Artifact,
    Backend,
    EventDataFrame,
    InSamplePredictions,
    Pipeline,
    PipelineCard,
    TrainedPipelineCard,
)
from ..events import _create_events
from ..splitters import Bounds, Fold, Splitter, get_splits
from ..utils.list import unpack_list_of_tuples, wrap_in_list
from .backend import get_backend
from .checks import check_types
from .common import _train_on_window
from .types import BackendType
from .utils import _extract_trained_pipelines
from .wrap import wrap_transformation_if_needed

logger = logging.getLogger("fold:loop")


def train(
    pipelinecard: PipelineCard | Pipeline,
    X: pd.DataFrame | None,
    y: pd.Series,
    splitter: Splitter,
    events: EventDataFrame | None = None,
    backend: BackendType | Backend | str = BackendType.no,
    silent: bool = False,
    for_deployment: bool = False,
) -> tuple[TrainedPipelineCard, Artifact, InSamplePredictions]:
    """
    Trains a pipeline on a given dataset, for all folds returned by the Splitter.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline to be fitted.
    X: pd.DataFrame | None
        Exogenous Data.
    y: pd.Series
        Endogenous Data (Target).
    splitter: Splitter
        Defines how the folds should be constructed.
    backend: str, BackendType = BackendType.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    events: EventDataFrame, optional = None
        Events that should be passed into the pipeline, by default None.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    return_artifacts: bool = False
        Whether to return the artifacts of the training process, by default False.
    return_insample: bool = False
        Whether to return the in-sample predictions of the training process, by default False.
    for_deployment: bool = False
        Whether the pipeline is being trained for deployment, meaning it'll only have the last fold, by default False.

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
    if events is None:
        events = _create_events(
            y,
            event_filter=pipelinecard.event_filter,
            labeler=pipelinecard.event_labeler,
        )
    if events is not None and events.shape[0] != X.shape[0]:
        logger.warning("The number of events does not match the number of samples.")
        events = events.reindex(X.index)
    artifact = Artifact.from_events(X.index, events)
    backend = get_backend(backend)

    preprocessing_max_memory_size = (
        get_maximum_memory_size(pipelinecard.preprocessing)
        if pipelinecard.preprocessing
        else 0
    )

    if pipelinecard.preprocessing is not None:
        preprocessing_pipeline = wrap_in_list(pipelinecard.preprocessing)
        preprocessing_pipeline = wrap_transformation_if_needed(preprocessing_pipeline)
        (
            _,
            trained_preprocessing_pipeline,
            preprocessed_X,
            preprocessed_artifact,
        ) = _train_on_window(
            X=X,
            y=y,
            artifact=artifact,
            pipeline=preprocessing_pipeline,
            split=Fold(
                index=0,
                train_bounds=[Bounds(0, len(X))],
                test_bounds=[Bounds(0, len(X))],
            ),
            backend=backend,
            project_name=f"{pipelinecard.project_name}-Preprocessing"
            if pipelinecard.project_name is not None
            else "Preprocessing",
            show_progress=True,
            project_hyperparameters=pipelinecard.project_hyperparameters,
            preprocessing_max_memory_size=0,
        )
        assert preprocessed_X.shape[0] == X.shape[0]
        assert preprocessed_artifact.shape[0] == artifact.shape[0]
        X = preprocessed_X
        artifact = preprocessed_artifact

        if pipelinecard.trim_initial_period_after_preprocessing:
            X = X.iloc[preprocessing_max_memory_size:]
            y = y.iloc[preprocessing_max_memory_size:]
            artifact = artifact.iloc[preprocessing_max_memory_size:]
            events = events[X.index[0] :]

    pipeline = wrap_in_list(pipelinecard.pipeline)
    pipeline = wrap_transformation_if_needed(pipeline)

    splits = get_splits(
        splitter=splitter,
        index=X.index,
        gap_before=pipelinecard.project_hyperparameters["forward_horizon"]
        if pipelinecard.project_hyperparameters
        else 0,
        gap_after=preprocessing_max_memory_size,
    )
    if len(splits) == 0:
        raise ValueError("No splits were generated by the Splitter.")

    if for_deployment:
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
                splits[-1:],
                backend,
                f"{pipelinecard.project_name}-Pipeline",
                pipelinecard.project_hyperparameters,
                preprocessing_max_memory_size,
                not silent,
            )
        )
    else:
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
                backend,
                f"{pipelinecard.project_name}-Pipeline",
                pipelinecard.project_hyperparameters,
                preprocessing_max_memory_size,
                not silent,
            )
        )

    trained_pipelines = TrainedPipelineCard(
        project_name=pipelinecard.project_name or "",
        project_hyperparameters=pipelinecard.project_hyperparameters,
        preprocessing=[pd.Series(p, index=[0]) for p in trained_preprocessing_pipeline]
        if pipelinecard.preprocessing
        else None,
        pipeline=_extract_trained_pipelines(processed_idx, processed_pipelines),
        event_labeler=pipelinecard.event_labeler,
        event_filter=pipelinecard.event_filter,
        trim_initial_period_after_preprocessing=pipelinecard.trim_initial_period_after_preprocessing,
    )
    processed_artifacts = concat_on_index_without_duplicates(processed_artifacts)
    processed_predictions = concat_on_index_without_duplicates(processed_predictions)
    assert processed_artifacts.index.equals(processed_predictions.index)
    return (
        trained_pipelines,
        processed_artifacts,
        processed_predictions,
    )
