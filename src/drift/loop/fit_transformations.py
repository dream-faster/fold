from typing import List

import pandas as pd
from tqdm import tqdm

from ..all_types import TransformationsOverTime
from ..transformations.base import Transformation
from ..utils.splitters import Split, Splitter


def fit_transformations(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
    transformations: List[Transformation],
) -> TransformationsOverTime:

    # easy to parallize this with ray
    processed_transformations = [
        __process_transformations_window(X, y, transformations, split)
        for split in tqdm(splitter.splits())
    ]

    idx, only_transformations = zip(*processed_transformations)

    transformations_over_time = [
        pd.Series(
            transformation_over_time,
            index=idx,
            name=transformation_over_time[0].name,
        )
        for transformation_over_time in zip(*only_transformations)
    ]

    return transformations_over_time


def __process_transformations_window(
    X: pd.DataFrame,
    y: pd.Series,
    transformations: list[Transformation],
    split: Split,
) -> tuple[int, list[Transformation]]:

    X_train = X.iloc[split.train_window_start : split.train_window_end]
    y_train = y.iloc[split.train_window_start : split.train_window_end]

    current_transformations = [t.clone() for t in transformations]
    for transformation in current_transformations:
        X_train = transformation.fit_transform(X_train, y_train)

    return split.model_index, current_transformations
