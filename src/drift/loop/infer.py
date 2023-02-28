from typing import List

import pandas as pd

from ..all_types import TransformationsOverTime
from ..transformations.base import Composite, Transformation, Transformations
from .common import recursively_transform

DeployableTransformations = Transformations


def to_deployable_transformations(
    transformations_over_time: TransformationsOverTime,
) -> DeployableTransformations:
    return [
        transformation_over_time.loc[-1]
        for transformation_over_time in transformations_over_time
    ]


def infer(
    transformations_over_time: DeployableTransformations,
    past_X: pd.DataFrame,
    X: pd.DataFrame,
) -> OutSamplePredictions:

    X_test = pd.concat([past_X, X], axis="index")
    X_test = recursively_transform(X_test, current_transformations)
    outofsample_values = zip(*results)

    outofsample_predictions = pd.concat(outofsample_values, axis="index").squeeze()
    return outofsample_predictions


def update(
    transformations_over_time: DeployableTransformations, X: pd.DataFrame, y: pd.Series
):
    return recursively_transform(X, transformations_over_time)
