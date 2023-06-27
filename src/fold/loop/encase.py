# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import pandas as pd

from fold.base.classes import InSamplePredictions
from fold.base.scoring import score_results

from ..base import (
    Artifact,
    EventDataFrame,
    OutOfSamplePredictions,
    Pipeline,
    TrainedPipelines,
)
from ..splitters import Splitter
from ..utils.dataframe import ResolutionStrategy, concat_on_columns_with_duplicates
from .backtesting import backtest
from .training import train
from .types import Backend, TrainMethod

if TYPE_CHECKING:
    from krisi import ScoreCard


def backtest_score(
    trained_pipelines: TrainedPipelines,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    backend: Union[str, Backend] = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    events: Optional[EventDataFrame] = None,
    silent: bool = False,
    return_artifacts: bool = False,
    krisi_args: Optional[dict] = None,
) -> Union[
    Tuple["ScoreCard", OutOfSamplePredictions],
    Tuple["ScoreCard", OutOfSamplePredictions, Artifact],
]:
    """
    Run backtest then scoring.
    [`krisi`](https://github.com/dream-faster/krisi) is required to be installed.

    Parameters
    ----------
    trained_pipelines: TrainedPipelines
        The fitted pipelines, for all folds.
    X: Optional[pd.DataFrame]
        Exogenous Data.
    y: pd.Series
        Endogenous Data (Target).
    splitter: Splitter
        Defines how the folds should be constructed.
    backend: str, Backend = Backend.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    sample_weights: Optional[pd.Series] = None
        Weights assigned to each sample/timestamp, that are passed into models that support it, by default None.
    events: EventDataFrame, optional = None
        Events that should be passed into the pipeline, by default None.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    return_artifacts: bool = False
        Whether to return the artifacts of the training process, by default False.
    krisi_args: Optional[Dict[str, Any]] = None
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
        sample_weights=sample_weights,
        events=events,
        silent=silent,
        return_artifacts=True,
    )

    scorecard = score_results(
        pred,
        y,
        artifacts=artifacts,
        krisi_args=krisi_args,
    )
    if return_artifacts:
        return scorecard, pred, artifacts
    else:
        return scorecard, pred


def train_backtest(
    pipeline: Pipeline,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    backend: Union[Backend, str] = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    events: Optional[EventDataFrame] = None,
    train_method: Union[TrainMethod, str] = TrainMethod.parallel,
    silent: bool = False,
    return_artifacts: bool = False,
    return_insample: bool = False,
) -> Union[
    Tuple[OutOfSamplePredictions, TrainedPipelines],
    Tuple[OutOfSamplePredictions, TrainedPipelines, Artifact],
    Tuple[OutOfSamplePredictions, TrainedPipelines, Artifact, InSamplePredictions],
]:
    """
    Run train and backtest.

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
    backend: str, Backend = Backend.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    sample_weights: Optional[pd.Series] = None
        Weights assigned to each sample/timestamp, that are passed into models that support it, by default None.
    events: EventDataFrame, optional = None
        Events that should be passed into the pipeline, by default None.
    train_method: TrainMethod, str = TrainMethod.parallel
        The training methodology, by default `parallel`.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    return_artifacts: bool = False
        Whether to return the artifacts of the process, by default False.
    return_insample: bool = False

    Returns
    -------
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    TrainedPipelines
        The fitted pipelines, for all folds.
    """
    trained_pipelines, train_artifacts, insample_predictions = train(
        pipeline,
        X,
        y,
        splitter,
        sample_weights,
        events=events,
        train_method=train_method,
        backend=backend,
        silent=silent,
        return_artifacts=True,
        return_insample=True,
    )

    pred, backtest_artifacts = backtest(
        trained_pipelines,
        X,
        y,
        splitter,
        backend=backend,
        sample_weights=sample_weights,
        events=events,
        silent=silent,
        return_artifacts=True,
    )

    if return_artifacts:
        artifacts = concat_on_columns_with_duplicates(
            [train_artifacts, backtest_artifacts],
            strategy=ResolutionStrategy.last,
        )
        if return_insample:
            return pred, trained_pipelines, artifacts, insample_predictions
        else:
            return pred, trained_pipelines, artifacts
    else:
        if return_insample:
            return pred, trained_pipelines, insample_predictions
        else:
            return pred, trained_pipelines


def train_evaluate(
    pipeline: Pipeline,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    backend: Union[str, Backend] = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    events: Optional[EventDataFrame] = None,
    train_method: Union[TrainMethod, str] = TrainMethod.parallel,
    silent: bool = False,
    return_artifacts: bool = False,
    krisi_args: Optional[Dict[str, Any]] = None,
) -> Union[
    Tuple[
        "ScoreCard",
        OutOfSamplePredictions,
        TrainedPipelines,
    ],
    Tuple[
        "ScoreCard",
        OutOfSamplePredictions,
        TrainedPipelines,
        Artifact,
    ],
]:
    """
    Run train, backtest then run scoring.
    [`krisi`](https://github.com/dream-faster/krisi) needs to be installed.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline to be fitted.
    X: pd.DataFrame, optional
        Exogenous Data.
    y: pd.Series
        Endogenous Data (Target).
    splitter: Splitter
        Defines how the folds should be constructed.
    backend: str, Backend = Backend.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    sample_weights: pd.Series, optional = None
        Weights assigned to each sample/timestamp, that are passed into models that support it, by default None.
    events: EventDataFrame, optional = None
        Events that should be passed into the pipeline, by default None.
    train_method: TrainMethod, str = TrainMethod.parallel
        The training methodology, by default `parallel`.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    krisi_args: Dict[str, Any], optional = None
        Arguments that will be passed into `krisi` score function, by default None.

    Returns
    -------
    "ScoreCard"
        A ScoreCard from `krisi`.
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    TrainedPipelines
        The fitted pipelines, for all folds.
    """
    trained_pipelines, train_artifacts = train(
        pipeline,
        X,
        y,
        splitter,
        sample_weights,
        events=events,
        train_method=train_method,
        backend=backend,
        silent=silent,
        return_artifacts=True,
    )

    scorecard, pred, backtest_artifacts = backtest_score(
        trained_pipelines,
        X,
        y,
        splitter,
        backend=backend,
        sample_weights=sample_weights,
        events=events,
        silent=silent,
        return_artifacts=True,
        krisi_args=krisi_args,
    )

    if return_artifacts:
        return (
            scorecard,
            pred,
            trained_pipelines,
            concat_on_columns_with_duplicates(
                [train_artifacts, backtest_artifacts], strategy=ResolutionStrategy.last
            ),
        )
    else:
        return scorecard, pred, trained_pipelines
