from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import pandas as pd

from fold.models.base import Model

from ..transformations.base import Composite, Transformation, Transformations
from ..utils.checks import is_prediction


def recursively_transform(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    transformations: Transformations,
    fit: bool,
    is_first_split: bool,
) -> pd.DataFrame:
    """
    The main function to transform (and fit) a pipline of transformations.
    is_first_split is used to determine whether to run the inner loop for models that have `requires_continuous_updating` set to True.
    """
    if isinstance(transformations, List):
        for transformation in transformations:
            X = recursively_transform(
                X, y, sample_weights, transformation, fit, is_first_split
            )
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
                is_first_split,
            )
            for index, child_transformation in enumerate(
                composite.get_child_transformations_primary()
            )
        ]

        if composite.properties.primary_only_single_pipeline:
            assert len(results_primary) == 1, ValueError(
                "Expected single output from primary transformations, got"
                f" {len(results_primary)} instead."
            )
        if composite.properties.primary_requires_predictions:
            assert is_prediction(results_primary[0]), ValueError(
                "Expected predictions from primary transformations, but got something"
                " else."
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
                    is_first_split,
                )
                for index, child_transformation in enumerate(secondary_transformations)
            ]

            if composite.properties.secondary_only_single_pipeline:
                assert len(results_secondary) == 1, ValueError(
                    "Expected single output from secondary transformations, got"
                    f" {len(results_secondary)} instead."
                )
            if composite.properties.secondary_requires_predictions:
                assert is_prediction(results_secondary[0]), ValueError(
                    "Expected predictions from secondary transformations, but got"
                    " something else."
                )

            return composite.postprocess_result_secondary(
                results_primary, results_secondary
            )

    elif isinstance(transformations, Transformation) or isinstance(
        transformations, Model
    ):
        if len(X) == 0:
            return pd.DataFrame()

        if transformations.properties.requires_continuous_updates and (
            (fit and not is_first_split) or not fit
        ):
            # If the transformation requires continuous updates, and this is the not first split, and we're in inference

            y_df = y.to_frame() if y is not None else None
            # We need to run the inference & fit loop on each row, sequentially (one-by-one).
            # This is so the transformation can update its parameters after each sample.

            # Important: depending on whether we're training or not:
            # - we call fit() _before_ transform(), at training time. The output are in-sample predictions.
            # - we call fit() after transform(), at backtesting/inference time. The output are then out-of-sample predictions.
            def transform_row_train(X_row, y_row, sample_weights_row):
                transformations.update(X_row, y_row, sample_weights_row)
                result = transformations.transform(X_row, in_sample=False)
                return result

            def transform_row_inference_backtest(X_row, y_row, sample_weights_row):
                result = transformations.transform(X_row, in_sample=False)
                if y_row is not None:
                    transformations.update(X_row, y_row, sample_weights_row)
                return result

            transform_row_function = (
                transform_row_train if fit else transform_row_inference_backtest
            )
            return pd.concat(
                [
                    transform_row_function(
                        X.loc[index:index],
                        y_df.loc[index] if y is not None else None,
                        sample_weights.loc[index]
                        if sample_weights is not None
                        else None,
                    )
                    for index in X.index
                ],
                axis="index",
            )

        else:
            if fit:
                transformations.fit(X, y, sample_weights)
            return transformations.transform(X, in_sample=fit)
    else:
        raise ValueError(
            f"{transformations} is not a Fold Transformation, but of type"
            f" {type(transformations)}"
        )


def deepcopy_transformations(transformation: Transformations) -> Transformations:
    if isinstance(transformation, List):
        return [deepcopy_transformations(t) for t in transformation]
    elif isinstance(transformation, Composite):
        return transformation.clone(deepcopy_transformations)
    else:
        return deepcopy(transformation)
