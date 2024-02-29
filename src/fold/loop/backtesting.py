# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


import logging

import pandas as pd
from finml_utils.dataframes import concat_on_index, trim_initial_nans

from ..base.classes import (
    Artifact,
    Backend,
    EventDataFrame,
    OutOfSamplePredictions,
    TrainedPipelineCard,
)
from ..base.utils import get_last_trained_pipeline, get_maximum_memory_size
from ..events import _create_events
from ..splitters import Bounds, Fold, Splitter, get_splits
from ..utils.list import unpack_list_of_tuples
from .backend import get_backend
from .checks import check_types
from .common import _backtest_on_window
from .types import BackendType

logger = logging.getLogger("fold:loop")


def backtest(
    trained_pipelinecard: TrainedPipelineCard,
    X: pd.DataFrame | None,
    y: pd.Series,
    splitter: Splitter,
    backend: BackendType | Backend | str = BackendType.no,
    events: EventDataFrame | None = None,
    silent: bool = False,
    mutate: bool = False,
    return_artifacts: bool = False,
) -> OutOfSamplePredictions | tuple[OutOfSamplePredictions, Artifact]:
    """
    Run backtest on TrainedPipelineCard and given data.

    Parameters
    ----------

    trained_pipelines: TrainedPipelineCard
        The fitted pipelines, for all folds.
    X: pd.DataFrame, optional
        Exogenous Data.
    y: pd.Series
        Endogenous Data (Target).
    splitter: Splitter
        Defines how the folds should be constructed.
    backend: str, BackendType = BackendType.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    sample_weights: pd.Series, optional = None
        Weights assigned to each sample/timestamp, that are passed into models that support it, by default None.
    events: EventDataFrame, optional = None
        Events that should be passed into the pipeline, by default None.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    mutate: bool = False
        Whether `trained_pipelines` should be mutated, by default False. This is discouraged.
    return_artifacts: bool = False
        Whether to return the artifacts of the backtesting process, by default False.

    Returns
    -------
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    """
    backend = get_backend(backend)
    X, y = check_types(X, y)
    if events is None:
        events = _create_events(
            y,
            event_filter=trained_pipelinecard.event_filter,
            labeler=trained_pipelinecard.event_labeler,
        )
    if events is not None and events.shape[0] != X.shape[0]:
        logger.warning("The number of events does not match the number of samples.")
        events = events.reindex(X.index)
    artifact = Artifact.from_events(X.index, events)

    preprocessing_max_memory_size = (
        get_maximum_memory_size(
            get_last_trained_pipeline(trained_pipelinecard.preprocessing)
        )
        if trained_pipelinecard.preprocessing
        else 0
    )

    if trained_pipelinecard.preprocessing is not None:
        preprocessed_X, preprocessed_artifacts = _backtest_on_window(
            trained_pipelines=trained_pipelinecard.preprocessing,
            split=Fold(
                index=0,
                train_bounds=[Bounds(0, len(X))],
                test_bounds=[Bounds(0, len(X))],
            ),
            X=X,
            y=y,
            artifact=artifact,
            backend=backend,
            mutate=mutate,
        )

        assert preprocessed_X.shape[0] == X.shape[0]
        assert preprocessed_artifacts.shape[0] == artifact.shape[0]
        X = preprocessed_X
        artifact = preprocessed_artifacts

        if trained_pipelinecard.trim_initial_period_after_preprocessing:
            X = X.iloc[preprocessing_max_memory_size:]
            y = y.iloc[preprocessing_max_memory_size:]
            artifact = artifact.iloc[preprocessing_max_memory_size:]
            events = events[X.index[0] :]

    results, artifacts = unpack_list_of_tuples(
        backend.backtest_pipeline(
            _backtest_on_window,
            trained_pipelinecard.pipeline,
            get_splits(
                splitter=splitter,
                index=X.index,
                gap_before=trained_pipelinecard.project_hyperparameters[
                    "forward_horizon"
                ]
                if trained_pipelinecard.project_hyperparameters
                else 0,
                gap_after=preprocessing_max_memory_size,
            ),
            X,
            y,
            artifact,
            backend,
            mutate=mutate,
            silent=silent,
        )
    )
    results = trim_initial_nans(concat_on_index(results))
    if return_artifacts:
        return results, concat_on_index(artifacts)
    return results
