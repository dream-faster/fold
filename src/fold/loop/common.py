from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import List, Optional

import pandas as pd

from fold.models.base import Model
from fold.utils.pandas import trim_initial_nans

from ..transformations.base import Composite, Transformation, Transformations
from ..utils.checks import is_prediction


class Stage(Enum):
    inital_fit = "inital_fit"
    update = "update"
    infer = "infer"

    def is_fit_or_update(self) -> bool:
        return self in [Stage.inital_fit, Stage.update]


def recursively_transform(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    transformations: Transformations,
    stage: Stage,
) -> pd.DataFrame:
    """
    The main function to transform (and fit or update) a pipline of transformations.
    `stage` is used to determine whether to run the inner loop for online models.
    """
    if isinstance(transformations, List):
        for transformation in transformations:
            X = recursively_transform(X, y, sample_weights, transformation, stage)
        return X

    elif isinstance(transformations, Composite):
        composite: Composite = transformations
        # TODO: here we have the potential to parallelize/distribute training of child transformations
        composite.before_fit(X)

        results_primary = [
            process_primary_child_transform(
                composite,
                index,
                child_transformation,
                X,
                y,
                sample_weights,
                stage,
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
            return composite.postprocess_result_primary(results_primary, y)
        else:
            results_secondary = [
                process_secondary_child_transform(
                    composite,
                    index,
                    child_transformation,
                    X,
                    y,
                    sample_weights,
                    results_primary,
                    stage,
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
                results_primary, results_secondary, y
            )

    elif isinstance(transformations, Transformation) or isinstance(
        transformations, Model
    ):
        if len(X) == 0:
            return pd.DataFrame()

        # If the transformation needs to be "online", and:
        # - we're training, and this is not the first split, or
        # - we're not training (i.e. we're backtesting or inferring)
        # enter the inner loop.
        if (
            transformations.properties.mode == Transformation.Properties.Mode.online
            and stage == Stage.update
        ):
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
                transform_row_train
                if stage == Stage.inital_fit
                else transform_row_inference_backtest
            )
            return pd.concat(
                [
                    transform_row_function(
                        X.loc[index:index],
                        y_df.loc[index:index] if y is not None else None,
                        sample_weights.loc[index]
                        if sample_weights is not None
                        else None,
                    )
                    for index in X.index
                ],
                axis="index",
            )

        else:
            X, y = trim_initial_nans(X, y)
            if stage == Stage.inital_fit:
                transformations.fit(X, y, sample_weights)
            elif stage == Stage.update:
                transformations.update(X, y, sample_weights)
            return transformations.transform(X, in_sample=stage == Stage.inital_fit)

    else:
        raise ValueError(
            f"{transformations} is not a Fold Transformation, but of type"
            f" {type(transformations)}"
        )


def process_primary_child_transform(
    composite: Composite,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    stage: Stage,
) -> pd.DataFrame:
    X, y = composite.preprocess_primary(X, index, y, fit=stage.is_fit_or_update())
    return recursively_transform(X, y, sample_weights, child_transform, stage)


def process_secondary_child_transform(
    composite: Composite,
    index: int,
    child_transform: Transformations,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    sample_weights: Optional[pd.Series],
    results_primary: List[pd.DataFrame],
    stage: Stage,
) -> pd.DataFrame:
    X, y = composite.preprocess_secondary(
        X, y, results_primary, index, fit=stage.is_fit_or_update()
    )
    return recursively_transform(X, y, sample_weights, child_transform, stage)


def deepcopy_transformations(transformation: Transformations) -> Transformations:
    if isinstance(transformation, List):
        return [deepcopy_transformations(t) for t in transformation]
    elif isinstance(transformation, Composite):
        return transformation.clone(deepcopy_transformations)
    else:
        return deepcopy(transformation)
