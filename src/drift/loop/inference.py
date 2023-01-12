from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..all_types import (
    InSamplePredictions,
    OutSamplePredictions,
    TransformationsOverTime,
)
from ..models.base import Model
from ..splitters import Split, Splitter
from ..transformations.base import Transformation


def infer(
    transformations_over_time: TransformationsOverTime,
    X: pd.DataFrame,
    splitter: Splitter,
    name: Optional[str] = None,
) -> tuple[InSamplePredictions, OutSamplePredictions]:

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
    X_train = X.iloc[split.train_window_start : split.train_window_end]

    current_transformations = [
        transformation_over_time.loc[split.model_index]
        for transformation_over_time in transformations_over_time
    ]
    for transformation in current_transformations:
        X_train = transformation.transform(X_train)

    X_test = X.iloc[split.test_window_start : split.test_window_end]
    for transformation in current_transformations:
        X_test = transformation.transform(X_test)

    return X_train, X_test
