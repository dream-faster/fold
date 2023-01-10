from copy import deepcopy
from typing import List, Union

import pandas as pd
from tqdm import tqdm

from ..all_types import TransformationsOverTime
from ..models.base import Model
from ..transformations.base import Transformation
from ..utils.splitters import Split, Splitter


def walk_forward_train(
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
    transformations: List[Union[Transformation, Model]],
    split: Split,
) -> tuple[int, List[Union[Transformation, Model]]]:

    X_train = X.iloc[split.train_window_start : split.train_window_end]
    y_train = y.iloc[split.train_window_start : split.train_window_end]

    current_transformations = [deepcopy(t) for t in transformations]
    for transformation in current_transformations:
        if isinstance(transformation, Transformation):
            X_train = transformation.fit_transform(X_train, y_train)
        elif isinstance(transformation, Model):
            transformation.fit(X_train, y_train)
            X_train = transformation.predict(X_train)
        else:
            raise ValueError(
                f"{type(transformation)} is not a Drift Model or Transformation."
            )

    return split.model_index, current_transformations
