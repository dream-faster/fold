from copy import deepcopy
from typing import Callable, List, Union

import pandas as pd
from sklearn.base import BaseEstimator
from tqdm import tqdm

from ..all_types import TransformationsOverTime
from ..models.base import Model
from ..splitters import Split, Splitter
from ..transformations.base import Composite, Transformation
from .process import process_pipeline


def train(
    transformations: List[Union[Transformation, Model, Callable, BaseEstimator]],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
) -> TransformationsOverTime:

    transformations = process_pipeline(transformations)

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
    transformations: List[Transformation],
    split: Split,
) -> tuple[int, List[Union[Transformation, Model]]]:

    X_train = X.iloc[split.train_window_start : split.train_window_end]
    y_train = y.iloc[split.train_window_start : split.train_window_end]

    transformations = [deepcopy_transformation(t) for t in transformations]

    for transformation in transformations:

        # TODO: here we have the potential to parallelize/distribute training of child transformations
        if isinstance(transformation, Composite):
            for child_transformation in transformation.get_child_transformations():
                child_transformation.fit(X_train, y_train)
        else:
            transformation.fit(X_train, y_train)

        X_train = transformation.transform(X_train)

    return split.model_index, transformations


def deepcopy_transformation(transformation: Transformation) -> Transformation:
    if isinstance(transformation, Composite):
        transformation.set_child_transformations(
            [
                deepcopy_transformation(child_transformation)
                for child_transformation in transformation.get_child_transformations()
            ]
        )
        return transformation
    else:
        return deepcopy(transformation)
