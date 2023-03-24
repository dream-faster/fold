from typing import TYPE_CHECKING, Tuple

import pandas as pd

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
    splitter: Splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2),
) -> Tuple["ScoreCard", OutOfSamplePredictions]:
    from krisi import score

    pred = backtest(transformations_over_time, X, y, splitter)
    scorecard = score(y[pred.index], pred.squeeze())

    return scorecard, pred


def train_backtest_score(
    transformations: BlocksOrWrappable,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2),
) -> Tuple["ScoreCard", OutOfSamplePredictions, TransformationsOverTime]:
    transformations_over_time = train(transformations, X, y, splitter)

    scorecard, pred = backtest_score(transformations_over_time, X, y, splitter)

    return scorecard, pred, transformations_over_time
