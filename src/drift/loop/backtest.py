from enum import Enum
from typing import Tuple, Union

import pandas as pd
from tqdm import tqdm

from ..all_types import (
    InSamplePredictions,
    OutOfSamplePredictions,
    TransformationsOverTime,
)
from ..splitters import Split, Splitter
from ..transformations.common import get_flat_list_of_transformations
from .common import deepcopy_transformations, recursively_transform


def backtest(
    transformations_over_time: TransformationsOverTime,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
) -> Tuple[InSamplePredictions, OutOfSamplePredictions]:
    """
    Backtest a list of transformations over time.
    Run backtest on a set of TransformationsOverTime and given data.
    Does not mutate or change the transformations in any way, aka you can backtest multiple times.
    """

    results = [
        __backtest_on_window(
            split,
            X,
            y,
            transformations_over_time,
        )
        for split in tqdm(splitter.splits(length=len(X)))
    ]
    insample_values, outofsample_values = zip(*results)

    insample_predictions = pd.concat(insample_values, axis="index")
    outofsample_predictions = pd.concat(outofsample_values, axis="index")
    return insample_predictions, outofsample_predictions


def __backtest_on_window(
    split: Split,
    X: pd.DataFrame,
    y: pd.Series,
    transformations_over_time: TransformationsOverTime,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    current_transformations = [
        transformation_over_time.loc[split.model_index]
        for transformation_over_time in transformations_over_time
    ]

    X_train = X.iloc[split.train_window_start : split.train_window_end]
    X_train = recursively_transform(
        X_train, None, None, current_transformations, fit=False
    )

    X_test = X.iloc[split.train_window_start : split.test_window_end]
    X_test = recursively_transform(
        X_test, None, None, deepcopy_transformations(current_transformations), fit=False
    )

    test_window_size = split.test_window_end - split.test_window_start
    X_test = X_test.iloc[-test_window_size:]

    return X_train, X_test
