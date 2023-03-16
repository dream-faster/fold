from typing import Optional

import pandas as pd

from ..all_types import OutOfSamplePredictions
from ..transformations.base import DeployableTransformations
from .common import deepcopy_transformations, recursively_transform
from .types import Backend, Stage


def infer(
    transformations: DeployableTransformations,
    X: Optional[pd.DataFrame],
) -> OutOfSamplePredictions:
    """
    Run inference on a set of Transformations and given data.
    Does not mutate or change the transformations in any way.
    A follow-up call to `update` is required to update the transformations, when the ground truth is available.
    If X is None it will predict one step ahead.
    """
    if X is None:
        X = pd.DataFrame([0], columns=[0])
    else:
        assert type(X) is pd.DataFrame, "X must be a pandas DataFrame."

    results = recursively_transform(
        X, None, None, transformations, stage=Stage.infer, backend=Backend.no
    )
    return results


def update(
    transformations: DeployableTransformations,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    sample_weights: Optional[pd.Series] = None,
) -> DeployableTransformations:
    """
    Update a set of Transformations with new data.
    Returns a new set of Transformations, does not mutate the original.
    """
    if X is None:
        X = pd.DataFrame(0, index=y.index, columns=[0])
    else:
        assert type(X) is pd.DataFrame, "X must be a pandas DataFrame."
    assert type(y) is pd.Series, "y must be a pandas Series."

    transformations = deepcopy_transformations(transformations)
    _ = recursively_transform(
        X, y, sample_weights, transformations, stage=Stage.update, backend=Backend.no
    )
    return transformations
