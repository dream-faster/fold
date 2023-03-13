from typing import Optional

import pandas as pd
from tqdm import tqdm

from fold.utils.pandas import trim_initial_nans_single

from ..all_types import OutOfSamplePredictions, TransformationsOverTime
from ..splitters import Fold, Splitter
from .common import deepcopy_transformations, recursively_transform


def backtest(
    transformations_over_time: TransformationsOverTime,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
    sample_weights: Optional[pd.Series] = None,
) -> OutOfSamplePredictions:
    """
    Backtest a list of transformations over time.
    Run backtest on a set of TransformationsOverTime and given data.
    Does not mutate or change the transformations in any way, aka you can backtest multiple times.
    """

    assert type(X) is pd.DataFrame, "X must be a pandas DataFrame."
    assert type(y) is pd.Series, "y must be a pandas Series."

    results = [
        __backtest_on_window(
            split,
            X,
            y,
            transformations_over_time,
            sample_weights,
        )
        for split in tqdm(splitter.splits(length=len(X)))
    ]
    outofsample_predictions = pd.concat(results, axis="index")
    return outofsample_predictions


def __backtest_on_window(
    split: Fold,
    X: pd.DataFrame,
    y: pd.Series,
    transformations_over_time: TransformationsOverTime,
    sample_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    current_transformations = [
        transformation_over_time.loc[split.model_index]
        for transformation_over_time in transformations_over_time
    ]

    X_test = X.iloc[split.test_window_start : split.test_window_end]
    y_test = y.iloc[split.test_window_start : split.test_window_end]
    sample_weights_test = (
        sample_weights.iloc[split.train_window_start : split.test_window_end]
        if sample_weights is not None
        else None
    )
    X_test = recursively_transform(
        X_test,
        y_test,
        sample_weights_test,
        deepcopy_transformations(
            current_transformations
        ),  # we deepcopy here to avoid mutating the continuously updated transformations
        fit=False,
        is_first_split=split.order == 0,
    )

    return trim_initial_nans_single(X_test)
