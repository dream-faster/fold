from typing import TYPE_CHECKING, Tuple

import pandas as pd
from backtesting import backtest
from splitters import Splitter
from training import train
from transformations.base import BlocksOrWrappable

from ..all_types import OutOfSamplePredictions, TransformationsOverTime

if TYPE_CHECKING:
    from krisi import ScoreCard


def backtest_evaluate(
    transformations_over_time: TransformationsOverTime,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
) -> Tuple["ScoreCard", OutOfSamplePredictions]:
    from krisi import score

    pred = backtest(transformations_over_time, X, y, splitter)
    scorecard = score(y[pred.index], pred.squeeze())

    return scorecard, pred


def train_backtest_evaluate(
    transformations: BlocksOrWrappable,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
) -> Tuple["ScoreCard", OutOfSamplePredictions, TransformationsOverTime]:
    transformations_over_time = train(transformations, X, y, splitter)

    scorecard, pred = backtest_evaluate(transformations_over_time, X, y, splitter)

    return scorecard, pred, transformations_over_time
