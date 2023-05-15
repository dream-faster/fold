# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


import importlib.util
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
from sklearn.metrics import mean_squared_error

from fold.utils.checks import (
    all_have_probabilities,
    get_prediction_column,
    get_probabilities_columns,
)

from ..base import OutOfSamplePredictions, Pipeline, TrainedPipelines
from ..splitters import Splitter
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
    silent: bool = False,
    krisi_args: Optional[Dict[str, Any]] = None,
    evaluation_func: Callable = mean_squared_error,
) -> Tuple[Union["ScoreCard", Dict[str, float]], OutOfSamplePredictions]:
    """
    Run backtest then scoring.
    If [`krisi`](https://github.com/dream-faster/krisi) is installed it will use it to generate a ScoreCard,
    otherwise it will run the `evaluation_func` passed in.

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
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    krisi_args: Optional[Dict[str, Any]] = None
        Arguments that will be passed into `krisi` score function, by default None.
    evaluation_func: Callable = mean_squared_error
        Function to evaluate with if `krisi` is not available, by default `mean_squared_error`.


    Returns
    -------
    "ScoreCard", Dict[str, float]
        A ScoreCard if `krisi` is available, else the result of the `evaluation_func` in a dict
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    """

    pred, artifacts = backtest(
        trained_pipelines,
        X,
        y,
        splitter,
        backend,
        sample_weights,
        silent,
        return_artifacts=True,
    )

    probabilities = (
        get_probabilities_columns(pred) if all_have_probabilities([pred]) else None
    )
    pred_point = get_prediction_column(pred)

    if artifacts is not None and "label" in artifacts.columns:
        y = artifacts["label"].reindex(pred.index).dropna()

    if len(y) != len(pred_point):
        probabilities = probabilities[: len(y)]
        pred_point = pred_point[: len(y)]

    if importlib.util.find_spec("krisi") is not None:
        from krisi import score

        scorecard = score(
            y=y[pred_point.index],
            predictions=pred_point,
            probabilities=probabilities,
            **(krisi_args if krisi_args is not None else {}),
        )
    else:
        pred_point = get_prediction_column(pred)
        scorecard = {
            evaluation_func.__class__.__name__: evaluation_func(
                y[pred_point.index], pred_point.squeeze()
            )
        }
    return scorecard, pred


def train_backtest(
    pipeline: Pipeline,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    backend: Union[Backend, str] = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    train_method: Union[TrainMethod, str] = TrainMethod.parallel,
    silent: bool = False,
) -> Tuple[OutOfSamplePredictions, TrainedPipelines]:
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
    train_method: TrainMethod, str = TrainMethod.parallel
        The training methodology, by default `parallel`.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.

    Returns
    -------
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    TrainedPipelines
        The fitted pipelines, for all folds.
    """
    trained_pipelines = train(
        pipeline, X, y, splitter, sample_weights, train_method, backend, silent
    )

    pred = backtest(
        trained_pipelines,
        X,
        y,
        splitter,
        backend,
        sample_weights,
        silent,
    )

    return pred, trained_pipelines


def train_evaluate(
    pipeline: Pipeline,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    backend: Backend = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    train_method: Union[TrainMethod, str] = TrainMethod.parallel,
    silent: bool = False,
    krisi_args: Optional[Dict[str, Any]] = None,
    evaluation_func: Callable = mean_squared_error,
) -> Tuple[
    Union["ScoreCard", Dict[str, float]],
    OutOfSamplePredictions,
    TrainedPipelines,
]:
    """
    Run train, backtest then run scoring.
    If [`krisi`](https://github.com/dream-faster/krisi) is installed it will use it to generate a ScoreCard,
    otherwise it will run the `evaluation_func` passed in.

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
    train_method: TrainMethod, str = TrainMethod.parallel
        The training methodology, by default `parallel`.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    krisi_args: Dict[str, Any], optional = None
        Arguments that will be passed into `krisi` score function, by default None.
    evaluation_func: Callable = mean_squared_error
        Function to evaluate with if `krisi` is not available, by default `mean_squared_error`.

    Returns
    -------
    "ScoreCard", Dict[str, float]
        A ScoreCard if `krisi` is available, else the result of the `evaluation_func` in a dict
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    TrainedPipelines
        The fitted pipelines, for all folds.
    """
    trained_pipelines = train(
        pipeline, X, y, splitter, sample_weights, train_method, backend, silent
    )

    scorecard, pred = backtest_score(
        trained_pipelines,
        X,
        y,
        splitter,
        backend,
        sample_weights,
        silent,
        krisi_args,
        evaluation_func,
    )

    return scorecard, pred, trained_pipelines
