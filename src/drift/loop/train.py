from copy import deepcopy
from typing import List, Union

import pandas as pd
from tqdm import tqdm

from ..all_types import TransformationsOverTime
from ..models.base import Model
from ..splitters import Split, Splitter
from ..transformations.base import Transformation


def train(
    transformations: List[Union[Transformation, Model]],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
) -> TransformationsOverTime:

    # easy to parallize this with ray
    processed_transformations = [
        __process_transformations_window(X, y, transformations, split)
        for split in tqdm(splitter.splits(length=len(y)))
    ]

    idx, only_transformations = zip(*processed_transformations)

    return [
        pd.Series(
            transformation_over_time,
            index=idx,
            name=transformation_over_time[0].name,
        )
        for transformation_over_time in zip(*only_transformations)
    ]


def __process_transformations_window(
    X: pd.DataFrame,
    y: pd.Series,
    transformations: List[Union[Transformation, Model]],
    split: Split,
) -> tuple[int, List[Union[Transformation, Model]]]:

    X_train = X.iloc[split.train_window_start : split.train_window_end]
    y_train = y.iloc[split.train_window_start : split.train_window_end]

    current_transformations = [deepcopy(t) for t in transformations]
    for transformation in current_transformations:

        # TODO: here we have the potential to parallelize/distribute training of child transformations
        child_transformations = transformation.get_child_transformations()
        if child_transformations is not None:
            for child_transformation in child_transformations:
                child_transformation.fit(X_train, y_train)

        X_train = transformation.fit_transform(X_train, y_train)

    return split.model_index, current_transformations
