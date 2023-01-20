from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from ..all_types import (
    InSamplePredictions,
    OutSamplePredictions,
    TransformationsOverTime,
)
from ..splitters import Split, Splitter
from ..transformations.base import Composite, Transformations


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
    current_transformations = [
        transformation_over_time.loc[split.model_index]
        for transformation_over_time in transformations_over_time
    ]

    X_train = X.iloc[split.train_window_start : split.train_window_end]
    X_train = recursively_transform(X_train, current_transformations)

    X_test = X.iloc[split.test_window_start : split.test_window_end]
    X_test = recursively_transform(X_test, current_transformations)

    return X_train, X_test


def recursively_transform(
    X: pd.DataFrame,
    transformations: Transformations,
) -> pd.DataFrame:

    if isinstance(transformations, List):
        for transformation in transformations:
            X = recursively_transform(X, transformation)
        return X

    elif isinstance(transformations, Composite):
        # TODO: here we have the potential to parallelize/distribute training of child transformations
        results = [
            recursively_transform(transformations.preprocess_X(X), child_transformation)
            for child_transformation in transformations.get_child_transformations()
        ]
        return transformations.postprocess_result(results)

    else:
        return transformations.transform(X)
