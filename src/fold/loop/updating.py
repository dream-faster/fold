from typing import Optional

import pandas as pd

from ..base import Artifact, DeployablePipeline, EventDataFrame
from .checks import check_types
from .common import deepcopy_pipelines, recursively_transform
from .types import Backend, Stage


def update(
    pipeline: DeployablePipeline,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    sample_weights: Optional[pd.Series] = None,
    events: Optional[EventDataFrame] = None,
) -> DeployablePipeline:
    """
    Update a set of Transformations with new data.
    Returns a new set of Transformations, does not mutate the original.
    """
    X, y = check_types(X, y)
    artifact = Artifact.from_events_sample_weights(X.index, events, sample_weights)

    transformations = deepcopy_pipelines(pipeline)
    _ = recursively_transform(
        X,
        y,
        artifact,
        transformations,
        stage=Stage.update,
        backend=Backend.no,
    )
    return transformations
