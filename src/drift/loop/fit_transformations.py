import pandas as pd
from all_types import TransformationsOverTime
from tqdm import tqdm
from transformations.base import Transformation
from utils.splitters import Split, Splitter
from typing import List


def fit_transformations(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
    transformations: List[Transformation],
) -> TransformationsOverTime:
    transformations_over_time = [
        pd.Series(index=y.index, dtype="object").rename(t.name) for t in transformations
    ]

    # easy to parallize this with ray
    processed_transformations = [
        __process_transformations_window(X, y, transformations, split)
        for split in tqdm(splitter.splits())
    ]

    # aggregate processed transformations
    for index, transformation in processed_transformations:
        for transformation_index, transformation in enumerate(transformation):
            transformations_over_time[transformation_index].iloc[index] = transformation

    return transformations_over_time


def __process_transformations_window(
    X: pd.DataFrame,
    y: pd.Series,
    transformations: list[Transformation],
    split: Split,
) -> tuple[int, list[Transformation]]:

    X_train = X[split.train_window_start : split.train_window_end]
    y_train = y[split.train_window_start : split.train_window_end]

    current_transformations = [t.clone() for t in transformations]
    for transformation in current_transformations:
        X_train = transformation.fit_transform(X_train, y_train)

    return split.model_index, current_transformations
