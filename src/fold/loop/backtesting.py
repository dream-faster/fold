# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Optional, Tuple, Union

import pandas as pd

from fold.base.classes import TrainedPipelineCard

from ..base import Artifact, EventDataFrame, OutOfSamplePredictions
from ..splitters import Fold, Splitter
from ..utils.dataframe import concat_on_index
from ..utils.list import unpack_list_of_tuples
from ..utils.trim import trim_initial_nans, trim_initial_nans_single
from .backend import get_backend
from .checks import check_types
from .common import _backtest_on_window
from .types import Backend, BackendType


def backtest(
    trained_pipelinecard: TrainedPipelineCard,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    backend: Union[BackendType, Backend, str] = BackendType.no,
    sample_weights: Optional[pd.Series] = None,
    events: Optional[EventDataFrame] = None,
    silent: bool = False,
    mutate: bool = False,
    return_artifacts: bool = False,
) -> Union[OutOfSamplePredictions, Tuple[OutOfSamplePredictions, Artifact]]:
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
    artifact = Artifact.from_events_sample_weights(X.index, events, sample_weights)
    X, y, artifact = trim_initial_nans(X, y, artifact)

    if trained_pipelinecard.preprocessing is not None:
        preprocessed_X, preprocessed_artifacts = _backtest_on_window(
            trained_pipelinecard.preprocessing,
            Fold(0, 0, 0, len(X), 0, 0, 0, len(X)),
            X,
            y,
            artifact,
            backend,
            mutate=mutate,
        )

        assert preprocessed_X.shape[0] == X.shape[0]
        assert preprocessed_artifacts.shape[0] == artifact.shape[0]
        X = preprocessed_X
        artifact = preprocessed_artifacts

    results, artifacts = unpack_list_of_tuples(
        backend.backtest_pipeline(
            _backtest_on_window,
            trained_pipelinecard.pipeline,
            splitter.splits(index=X.index),
            X,
            y,
            artifact,
            backend,
            mutate=mutate,
            silent=silent,
        )
    )
    results = trim_initial_nans_single(concat_on_index(results, copy=False))
    if return_artifacts:
        return results, concat_on_index(artifacts, copy=False)
    else:
        return results
