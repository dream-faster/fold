from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Union

import pandas as pd

from drift.all_types import TransformationsOverTime

from ..transformations.base import Composite, Transformation, Transformations
from ..utils.checks import is_prediction


def recursively_transform(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    transformations: Transformations,
    fit: bool = False,
) -> pd.DataFrame:

    if isinstance(transformations, List):
        for transformation in transformations:
            X = recursively_transform(X, y, sample_weights, transformation, fit)
        return X

    elif isinstance(transformations, Composite):
        composite: Composite = transformations
        # TODO: here we have the potential to parallelize/distribute training of child transformations
        composite.before_fit(X)
        results_primary = [
            recursively_transform(
                composite.preprocess_X_primary(X, index, y),
                composite.preprocess_y_primary(y) if y is not None else None,
                sample_weights,
                child_transformation,
                fit,
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
                recursively_transform(
                    composite.preprocess_X_secondary(X, results_primary, index),
                    composite.preprocess_y_secondary(y, results_primary)
                    if y is not None
                    else None,
                    sample_weights,
                    child_transformation,
                    fit,
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

    elif isinstance(transformations, Transformation):
        if len(X) == 0:
            return pd.DataFrame()

        if transformations.properties.requires_continuous_updates:
            # If the transformation requires continuous updates, we need to run the inference
            # & fit loop on each row, sequentially (one-by-one).
            # This is so the transformation can update its parameters after each sample.
            def transform_fit_row(row):
                # Important: here, we do inference first, then we fit the transformation.
                # This is because we want to use the predictions _before_ the transformation has seen the data.
                result = transformations.transform(row)
                if fit:
                    transformations.fit(row, y, sample_weights)
                return result

            return X.apply(transform_fit_row, axis="index")

        else:
            if fit:
                transformations.fit(X, y, sample_weights)
            return transformations.transform(X)
    else:
        raise ValueError(
            f"{transformations} is not a Drift Transformation, but of type {type(transformations)}"
        )


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
