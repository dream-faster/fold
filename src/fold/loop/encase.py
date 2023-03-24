from typing import TYPE_CHECKING, Callable, Dict, Tuple, Union

import pandas as pd
from sklearn.metrics import mean_squared_error

from ..all_types import OutOfSamplePredictions, TransformationsOverTime
from ..splitters import ExpandingWindowSplitter, Splitter
from ..transformations.base import BlocksOrWrappable
from .backtesting import backtest
from .training import train

if TYPE_CHECKING:
    from krisi import ScoreCard


def backtest_score(
    transformations_over_time: TransformationsOverTime,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
    with_krisi: bool = False,
    evaluation_func: Callable = mean_squared_error,
) -> Tuple[Union["ScoreCard", Dict[str, float]], OutOfSamplePredictions]:
    pred = backtest(transformations_over_time, X, y, splitter)
    if with_krisi:
        from krisi import score

        scorecard = score(y[pred.index], pred.squeeze())
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
    with_krisi: bool = False,
) -> Tuple[
    Union["ScoreCard", Dict[str, float]],
    OutOfSamplePredictions,
    TransformationsOverTime,
]:
    transformations_over_time = train(transformations, X, y, splitter)

    scorecard, pred = backtest_score(
        transformations_over_time, X, y, splitter, with_krisi
    )

    return scorecard, pred, transformations_over_time
