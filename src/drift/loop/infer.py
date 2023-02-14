from typing import List

import pandas as pd

from ..all_types import TransformationsOverTime
from ..transformations.base import Composite, Transformations

DeployableTransformations = Transformations


def to_deployable_transformations(
    transformations_over_time: TransformationsOverTime,
) -> DeployableTransformations:
    return [
        transformation_over_time.loc[0]
        for transformation_over_time in transformations_over_time
    ]


# def infer(
#     transformations_over_time: DeployableTransformations,
#     X: pd.DataFrame,
# ) -> OutSamplePredictions:

#     X_test = X.iloc[split.test_window_start : split.test_window_end]
#     X_test = recursively_transform(X_test, current_transformations)
#     outofsample_values = zip(*results)

#     outofsample_predictions = pd.concat(outofsample_values).squeeze()
#     return outofsample_predictions


def recursively_transform(
    X: pd.DataFrame,
    transformations: Transformations,
) -> pd.DataFrame:

    if isinstance(transformations, List):
        for transformation in transformations:
            X = recursively_transform(X, transformation)
        return X

    elif isinstance(transformations, Composite):
        # TODO: here we have the potential to parallelize/distribute training of child transformations
        results_primary = [
            recursively_transform(
                transformations.preprocess_X_primary(X, index),
                child_transformation,
            )
            for index, child_transformation in enumerate(
                transformations.get_child_transformations_primary()
            )
        ]
        secondary_transformations = (
            transformations.get_child_transformations_secondary()
        )

        if secondary_transformations is None:
            return transformations.postprocess_result_primary(results_primary)
        else:
            results_secondary = [
                recursively_transform(
                    transformations.preprocess_X_secondary(X, results_primary, index),
                    child_transformation,
                )
                for index, child_transformation in enumerate(secondary_transformations)
            ]
            return transformations.postprocess_result_secondary(
                results_primary, results_secondary
            )

    else:
        if len(X) == 0:
            return pd.DataFrame()
        return transformations.transform(X)
