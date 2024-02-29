# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from finml_utils.dataframes import concat_on_index_without_duplicates

from ..base import (
    Artifact,
    Backend,
    EventDataFrame,
    InSamplePredictions,
    OutOfSamplePredictions,
    Pipeline,
    PipelineCard,
    TrainedPipelineCard,
)
from ..base.scoring import score_results
from ..splitters import Splitter
from .backtesting import backtest
from .training import train
from .types import BackendType

if TYPE_CHECKING:
    from krisi import ScoreCard


def backtest_score(
    trained_pipelines: TrainedPipelineCard,
    X: pd.DataFrame | None,
    y: pd.Series,
    splitter: Splitter,
    backend: BackendType | Backend | str = BackendType.no,
    events: EventDataFrame | None = None,
    silent: bool = False,
    return_artifacts: bool = False,
    krisi_args: dict | None = None,
) -> (
    tuple[ScoreCard, OutOfSamplePredictions]
    | tuple[ScoreCard, OutOfSamplePredictions, Artifact]
):
    """
    Run backtest then scoring.
    [`krisi`](https://github.com/dream-faster/krisi) is required to be installed.

    Parameters
    ----------
    trained_pipelines: TrainedPipelineCard
        The fitted pipelines, for all folds.
    X: pd.DataFrame | None
        Exogenous Data.
    y: pd.Series
        Endogenous Data (Target).
    splitter: Splitter
        Defines how the folds should be constructed.
    backend: str, Backend = Backend.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    events: EventDataFrame, optional = None
        Events that should be passed into the pipeline, by default None.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    return_artifacts: bool = False
        Whether to return the artifacts of the training process, by default False.
    krisi_args: dict | None = None
        Arguments that will be passed into `krisi` score function, by default None.

    Returns
    -------
    "ScoreCard"
        A ScoreCard from `krisi`.
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    """

    pred, artifacts = backtest(
        trained_pipelines,
        X,
        y,
        splitter,
        backend=backend,
        events=events,
        silent=silent,
        return_artifacts=True,
    )

    scorecard, pred = score_results(
        pred,
        y,
        artifacts=artifacts,
        krisi_args=krisi_args,
    )
    if return_artifacts:
        return scorecard, pred, artifacts
    return scorecard, pred


def train_backtest(
    pipeline: Pipeline | PipelineCard,
    X: pd.DataFrame | None,
    y: pd.Series,
    splitter: Splitter,
    backend: BackendType | Backend | str = BackendType.no,
    events: EventDataFrame | None = None,
    silent: bool = False,
) -> tuple[OutOfSamplePredictions, TrainedPipelineCard, Artifact, InSamplePredictions]:
    """
    Run train and backtest.

    Parameters
    ----------
    pipeline: Union[Pipeline, PipelineCard]
        The pipeline to be fitted.
    X: pd.DataFrame | None
        Exogenous Data.
    y: pd.Series
        Endogenous Data (Target).
    splitter: Splitter
        Defines how the folds should be constructed.
    backend: str, Backend = Backend.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    events: EventDataFrame, optional = None
        Events that should be passed into the pipeline, by default None.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    return_artifacts: bool = False
        Whether to return the artifacts of the process, by default False.
    return_insample: bool = False

    Returns
    -------
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    TrainedPipelineCard
        The fitted pipelines, for all folds.
    """
    trained_pipelines, train_artifacts, insample_predictions = train(
        pipeline,
        X,
        y,
        splitter,
        events=events,
        backend=backend,
        silent=silent,
    )

    pred, backtest_artifacts = backtest(
        trained_pipelines,
        X,
        y,
        splitter,
        backend=backend,
        events=events,
        silent=silent,
        return_artifacts=True,
    )

    artifacts = concat_on_index_without_duplicates(
        [train_artifacts, backtest_artifacts],
    )
    assert artifacts.index.is_monotonic_increasing
    return pred, trained_pipelines, artifacts, insample_predictions


def train_evaluate(
    pipeline: Pipeline | PipelineCard,
    X: pd.DataFrame | None,
    y: pd.Series,
    splitter: Splitter,
    backend: BackendType | Backend | str = BackendType.no,
    events: EventDataFrame | None = None,
    silent: bool = False,
    krisi_args: dict | None = None,
) -> tuple[
    ScoreCard,
    OutOfSamplePredictions,
    TrainedPipelineCard,
    Artifact,
    ScoreCard,
]:
    """
    Run train, backtest then run scoring.
    [`krisi`](https://github.com/dream-faster/krisi) needs to be installed.

    Parameters
    ----------
    pipeline: Union[Pipeline, PipelineCard]
        The pipeline to be fitted.
    X: pd.DataFrame, optional
        Exogenous Data.
    y: pd.Series
        Endogenous Data (Target).
    splitter: Splitter
        Defines how the folds should be constructed.
    backend: str, Backend = Backend.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    events: EventDataFrame, optional = None
        Events that should be passed into the pipeline, by default None.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    krisi_args: dict, optional = None
        Arguments that will be passed into `krisi` score function, by default None.

    Returns
    -------
    "ScoreCard"
        A ScoreCard from `krisi`.
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    TrainedPipelineCard
        The fitted pipelines, for all folds.
    """
    trained_pipelines, train_artifacts, insample_predictions = train(
        pipeline,
        X,
        y,
        splitter,
        events=events,
        backend=backend,
        silent=silent,
    )

    scorecard, pred, backtest_artifacts = backtest_score(
        trained_pipelines,
        X,
        y,
        splitter,
        backend=backend,
        events=events,
        silent=silent,
        return_artifacts=True,
        krisi_args=dict(sample_type="validation", **(krisi_args or {})),
    )
    scorecard_insample, _ = score_results(
        insample_predictions,
        y,
        artifacts=train_artifacts,
        krisi_args=dict(sample_type="insample", **(krisi_args or {})),
    )

    return (
        scorecard,
        pred,
        trained_pipelines,
        concat_on_index_without_duplicates([train_artifacts, backtest_artifacts]),
        scorecard_insample,
    )
