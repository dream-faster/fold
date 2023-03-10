from typing import Optional

import pandas as pd

from ..all_types import OutOfSamplePredictions
from ..transformations.base import DeployableTransformations
from .common import deepcopy_transformations, recursively_transform


def infer(
    transformations: DeployableTransformations,
    X: pd.DataFrame,
) -> OutOfSamplePredictions:
    """
    Run inference on a set of Transformations and given data.
    Does not mutate or change the transformations in any way.
    A follow-up call to `update` is required to update the transformations, when the ground truth is available.
    """

    assert type(X) is pd.DataFrame, "X must be a pandas DataFrame."

    results = recursively_transform(
        X, None, None, transformations, fit=False, is_first_split=False
    )
    return results


def update(
    transformations: DeployableTransformations,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series] = None,
) -> DeployableTransformations:
    """
    Update a set of Transformations with new data.
    Returns a new set of Transformations, does not mutate the original.
    """

    assert type(X) is pd.DataFrame, "X must be a pandas DataFrame."
    assert type(y) is pd.Series, "y must be a pandas Series."

    transformations = deepcopy_transformations(transformations)
    _ = recursively_transform(
        X, y, sample_weights, transformations, fit=True, is_first_split=False
    )
    return transformations
