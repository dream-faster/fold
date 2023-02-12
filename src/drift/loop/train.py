from copy import deepcopy
from typing import Callable, List, Union

import pandas as pd
from sklearn.base import BaseEstimator

from drift.utils.checks import is_prediction
from drift.utils.list import wrap_in_list

from ..all_types import TransformationsOverTime
from ..models.base import Model
from ..splitters import Split, Splitter
from ..transformations.base import Composite, Transformation, Transformations
from .backend.sequential import process_transformations
from .convenience import replace_transformation_if_not_drift_native


def train(
    transformations: List[
        Union[Transformation, Composite, Model, Callable, BaseEstimator]
    ],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
) -> TransformationsOverTime:

    transformations = wrap_in_list(transformations)
    transformations: Transformations = replace_transformation_if_not_drift_native(
        transformations
    )

    splits = splitter.splits(length=len(y))
    processed = process_transformations(
        process_transformations_window, transformations, X, y, splits
    )

    idx, only_transformations = zip(*processed)

    return [
        pd.Series(
            transformation_over_time,
            index=idx,
            name=transformation_over_time[0].name,
        )
        for transformation_over_time in zip(*only_transformations)
    ]


def process_transformations_window(
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
    elif isinstance(transformation, Composite):
        return transformation.clone(deepcopy_transformations)
    else:
        return deepcopy(transformation)


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
        transformations.before_fit(X)
        results_primary = [
            recursively_fit_transform(
                transformations.preprocess_X_primary(X, index),
                transformations.preprocess_y_primary(y),
                child_transformation,
            )
            for index, child_transformation in enumerate(
                transformations.get_child_transformations_primary()
            )
        ]

        if transformations.properties.primary_only_single_pipeline:
            assert len(results_primary) == 1, ValueError(
                f"Expected single output from primary transformations, got {len(results_primary)} instead."
            )
        if transformations.properties.primary_requires_predictions:
            assert is_prediction(results_primary[0]), ValueError(
                "Expected predictions from primary transformations, but got something else."
            )

        secondary_transformations = (
            transformations.get_child_transformations_secondary()
        )
        if secondary_transformations is None:
            return transformations.postprocess_result_primary(results_primary)
        else:
            results_secondary = [
                recursively_fit_transform(
                    transformations.preprocess_X_secondary(X, results_primary, index),
                    transformations.preprocess_y_secondary(y, results_primary),
                    child_transformation,
                )
                for index, child_transformation in enumerate(secondary_transformations)
            ]

            if transformations.properties.secondary_only_single_pipeline:
                assert len(results_secondary) == 1, ValueError(
                    f"Expected single output from secondary transformations, got {len(results_secondary)} instead."
                )
            if transformations.properties.secondary_requires_predictions:
                assert is_prediction(results_secondary[0]), ValueError(
                    "Expected predictions from secondary transformations, but got something else."
                )

            return transformations.postprocess_result_secondary(
                results_primary, results_secondary
            )

    else:
        transformations.fit(X, y)
        return transformations.transform(X)
