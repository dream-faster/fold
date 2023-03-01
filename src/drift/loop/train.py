from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator

from ..all_types import TransformationsOverTime
from ..models.base import Model
from ..splitters import Split, Splitter
from ..transformations.base import Composite, Transformation, Transformations
from ..utils.checks import is_prediction
from ..utils.list import wrap_in_list
from .backend.ray import process_transformations as process_transformations_ray
from .backend.sequential import (
    process_transformations as process_transformations_sequential,
)
from .convenience import replace_transformation_if_not_drift_native


class Backend(Enum):
    no = "no"
    ray = "ray"

    @staticmethod
    def from_str(value: Union[str, Backend]) -> Backend:
        if isinstance(value, Backend):
            return value
        for strategy in Backend:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown Backend: {value}")


def train(
    transformations: List[
        Union[Transformation, Composite, Model, Callable, BaseEstimator]
    ],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
    sample_weights: Optional[pd.Series] = None,
    backend: Backend = Backend.no,
) -> TransformationsOverTime:
    """
    Train a list of transformations over time.
    """

    transformations = wrap_in_list(transformations)
    transformations: Transformations = replace_transformation_if_not_drift_native(
        transformations
    )

    splits = splitter.splits(length=len(y))
    process_function = process_transformations_sequential
    if backend == Backend.ray:
        process_function = process_transformations_ray
    processed = process_function(
        process_transformations_window,
        transformations,
        X,
        y,
        sample_weights,
        splits,
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
    sample_weights: Optional[pd.Series],
    transformations: List[Union[Transformation, Composite]],
    split: Split,
) -> Tuple[int, List[Union[Transformation, Composite]]]:

    X_train = X.iloc[split.train_window_start : split.train_window_end]
    y_train = y.iloc[split.train_window_start : split.train_window_end]

    sample_weights_train = (
        sample_weights.iloc[split.train_window_start : split.train_window_end]
        if sample_weights is not None
        else None
    )

    transformations = deepcopy_transformations(transformations)
    X_train = recursively_fit_transform(
        X_train, y_train, sample_weights_train, transformations
    )

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
    sample_weights: Optional[pd.Series],
    transformations: Transformations,
) -> pd.DataFrame:

    if isinstance(transformations, List):
        for transformation in transformations:
            X = recursively_fit_transform(X, y, sample_weights, transformation)
        return X

    elif isinstance(transformations, Composite):
        composite: Composite = transformations
        # TODO: here we have the potential to parallelize/distribute training of child transformations
        composite.before_fit(X)
        results_primary = [
            recursively_fit_transform(
                composite.preprocess_X_primary(X, index, y),
                composite.preprocess_y_primary(y),
                sample_weights,
                child_transformation,
            )
            for index, child_transformation in enumerate(
                composite.get_child_transformations_primary()
            )
        ]

        if composite.properties.primary_only_single_pipeline:
            assert len(results_primary) == 1, ValueError(
                f"Expected single output from primary transformations, got {len(results_primary)} instead."
            )
        if composite.properties.primary_requires_predictions:
            assert is_prediction(results_primary[0]), ValueError(
                "Expected predictions from primary transformations, but got something else."
            )

        secondary_transformations = composite.get_child_transformations_secondary()
        if secondary_transformations is None:
            return composite.postprocess_result_primary(results_primary)
        else:
            results_secondary = [
                recursively_fit_transform(
                    composite.preprocess_X_secondary(X, results_primary, index),
                    composite.preprocess_y_secondary(y, results_primary),
                    sample_weights,
                    child_transformation,
                )
                for index, child_transformation in enumerate(secondary_transformations)
            ]

            if composite.properties.secondary_only_single_pipeline:
                assert len(results_secondary) == 1, ValueError(
                    f"Expected single output from secondary transformations, got {len(results_secondary)} instead."
                )
            if composite.properties.secondary_requires_predictions:
                assert is_prediction(results_secondary[0]), ValueError(
                    "Expected predictions from secondary transformations, but got something else."
                )

            return composite.postprocess_result_secondary(
                results_primary, results_secondary
            )

    else:
        if len(X) == 0:
            return pd.DataFrame()
        transformations.fit(X, y, sample_weights)
        return transformations.transform(X)
