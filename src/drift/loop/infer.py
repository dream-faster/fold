from typing import Optional

import pandas as pd

from ..all_types import OutOfSamplePredictions
from ..transformations.base import DeployableTransformations, Transformations
from .common import (
    deepcopy_transformations,
    recursively_fit_transform,
    recursively_transform,
)


def infer(
    transformations: DeployableTransformations,
    past_X: pd.DataFrame,
    X: pd.DataFrame,
) -> OutOfSamplePredictions:
    """
    Run inference on a set of Transformations and given data.
    Does not mutate or change the transformations in any way.
    A follow-up call to `update` is required to update the transformations, when the ground truth is available.
    """

    X_test = pd.concat([past_X, X], axis="index")
    results = recursively_transform(X_test, transformations)
    return results


def update(
    transformations: DeployableTransformations,
    X: pd.DataFrame,
    past_X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series] = None,
) -> DeployableTransformations:
    """
    Update a set of Transformations with new data.
    Returns a new set of Transformations, does not mutate the original.
    """
    transformations = deepcopy_transformations(transformations)
    X = pd.concat([past_X, X], axis="index")
    _ = recursively_fit_transform(X, y, sample_weights, transformations)
    return transformations
