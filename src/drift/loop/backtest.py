from typing import Tuple

import pandas as pd
from tqdm import tqdm

from ..all_types import (
    InSamplePredictions,
    OutSamplePredictions,
    TransformationsOverTime,
)
from ..splitters import Split, Splitter
from .infer import recursively_transform


def backtest(
    transformations_over_time: TransformationsOverTime,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
) -> Tuple[InSamplePredictions, OutSamplePredictions]:
    """
    Backtest a list of transformations over time.
    """

    results = [
        __inference_from_window(
            split,
            X,
            transformations_over_time,
        )
        for split in tqdm(splitter.splits(length=len(X)))
    ]
    insample_values, outofsample_values = zip(*results)

    insample_predictions = pd.concat(insample_values).squeeze()
    outofsample_predictions = pd.concat(outofsample_values).squeeze()
    return insample_predictions, outofsample_predictions


def __inference_from_window(
    split: Split,
    X: pd.DataFrame,
    transformations_over_time: TransformationsOverTime,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    current_transformations = [
        transformation_over_time.loc[split.model_index]
        for transformation_over_time in transformations_over_time
    ]

    X_train = X.iloc[split.train_window_start : split.train_window_end]
    X_train = recursively_transform(X_train, current_transformations)

    X_test = X.iloc[split.train_window_start : split.test_window_end]
    X_test = recursively_transform(X_test, current_transformations)

    test_window_size = split.test_window_end - split.test_window_start
    X_test = X_test.iloc[-test_window_size:]

    return X_train, X_test
