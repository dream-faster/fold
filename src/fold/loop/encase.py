import importlib.util
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
from sklearn.metrics import mean_squared_error

from ..all_types import OutOfSamplePredictions, TransformationsOverTime
from ..splitters import ExpandingWindowSplitter, Splitter
from ..transformations.base import BlocksOrWrappable
from .backtesting import backtest
from .training import train
from .types import Backend, TrainMethod

if TYPE_CHECKING:
    from krisi import ScoreCard


def backtest_score(
    transformations_over_time: TransformationsOverTime,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
    backend: Backend = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    silent: bool = False,
    mutate: bool = False,
    krisi_args: Optional[Dict[str, Any]] = None,
    evaluation_func: Callable = mean_squared_error,
) -> Tuple[Union["ScoreCard", Dict[str, float]], OutOfSamplePredictions]:
    pred = backtest(
        transformations_over_time,
        X,
        y,
        splitter,
        backend,
        sample_weights,
        silent,
        mutate,
    )

    if importlib.util.find_spec("krisi") is not None:
        from krisi import score

        scorecard = score(
            y[pred.index],
            pred.squeeze(),
            **(krisi_args if krisi_args is not None else {}),
        )
    else:
        scorecard = {
            "mean_squared_error": evaluation_func(y[pred.index], pred.squeeze())
        }
    return scorecard, pred


def train_backtest_score(
    transformations: BlocksOrWrappable,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2),
    backend: Backend = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    train_method: TrainMethod = TrainMethod.parallel,
    silent: bool = False,
    mutate: bool = False,
    krisi_args: Optional[Dict[str, Any]] = None,
    evaluation_func: Callable = mean_squared_error,
) -> Tuple[
    Union["ScoreCard", Dict[str, float]],
    OutOfSamplePredictions,
    TransformationsOverTime,
]:
    transformations_over_time = train(
        transformations, X, y, splitter, sample_weights, train_method, backend, silent
    )

    scorecard, pred = backtest_score(
        transformations_over_time,
        X,
        y,
        splitter,
        backend,
        sample_weights,
        silent,
        mutate,
        krisi_args,
        evaluation_func,
    )

    return scorecard, pred, transformations_over_time
