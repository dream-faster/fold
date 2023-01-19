from copy import deepcopy
from typing import Callable, List, Union

import pandas as pd
from sklearn.base import BaseEstimator
from tqdm import tqdm

from ..all_types import TransformationsOverTime
from ..models.base import Model
from ..models.ensemble import Ensemble
from ..splitters import Split, Splitter
from ..transformations.base import Composite, Transformation, Transformations
from ..transformations.concat import Concat
from .convenience import process_pipeline


def train(
    transformations: List[
        Union[Transformation, Composite, Model, Callable, BaseEstimator]
    ],
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
) -> tuple[int, List[Union[Transformation, Composite, Model]]]:

    X_train = X.iloc[split.train_window_start : split.train_window_end]
    y_train = y.iloc[split.train_window_start : split.train_window_end]

    transformations = deepcopy_transformations(transformations)
    X_train = recursively_fit_transform(X_train, y_train, transformations)

    return split.model_index, transformations


def deepcopy_transformations(
    transformation: Union[
        Transformation, Composite, List[Union[Transformation, Composite]]
    ]
) -> Union[Transformation, Composite, List[Union[Transformation, Composite]]]:
    if isinstance(transformation, List):
        return [deepcopy_transformations(t) for t in transformation]
    elif isinstance(transformation, Concat):
        return Concat(
            [
                deepcopy_transformations(c)
                for c in transformation.get_child_transformations()
            ],
            if_duplicate_keep=transformation.if_duplicate_keep,
        )
    elif isinstance(transformation, Ensemble):
        return Ensemble(
            [
                deepcopy_transformations(c)
                for c in transformation.get_child_transformations()
            ]
        )
    else:
        return deepcopy(transformation)


# enables recursive execution
def recursively_fit_transform(
    X: pd.DataFrame,
    y: pd.Series,
    transformations: Transformations,
) -> pd.DataFrame:

    if isinstance(transformations, List):
        for transformation in transformations:
            X = recursively_fit_transform(X, y, transformation)
        return X

    elif isinstance(transformations, Composite):
        # TODO: here we have the potential to parallelize/distribute training of child transformations
        results = [
            recursively_fit_transform(X, y, child_transformation)
            for child_transformation in transformations.get_child_transformations()
        ]
        return transformations.postprocess_result(results)

    else:
        transformations.fit(X, y)
        return transformations.transform(X)
