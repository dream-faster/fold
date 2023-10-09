from typing import Optional

import pandas as pd

from ..base import Artifact, EventDataFrame, TrainedPipelineCard
from ..utils.trim import trim_initial_nans
from .backend import get_backend
from .checks import check_types
from .common import deepcopy_pipelines, recursively_transform
from .types import BackendType, Stage


def update(
    pipelinecard: TrainedPipelineCard,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    events: Optional[EventDataFrame] = None,
) -> TrainedPipelineCard:
    """
    Update a set of Transformations with new data.
    Returns a new set of Transformations, does not mutate the original.
    """
    X, y = check_types(X, y)
    artifact = Artifact.from_events(X.index, events)
    X, y, artifact = trim_initial_nans(X, y, artifact)
    backend = get_backend(BackendType.no)

    if pipelinecard.preprocessing is not None:
        preprocessing_pipelines, X, preprocessing_artifact = recursively_transform(
            X,
            None,
            artifact,
            deepcopy_pipelines(pipelinecard.preprocessing),
            stage=Stage.update,
            backend=backend,
        )

    pipeline, result, artifact = recursively_transform(
        X,
        y,
        artifact,
        deepcopy_pipelines(pipelinecard.pipeline),
        stage=Stage.update,
        backend=backend,
    )
    return TrainedPipelineCard(
        name=pipelinecard.name,
        preprocessing=preprocessing_pipelines,
        pipeline=pipeline,
        event_filter=pipelinecard.event_filter,
        event_labeler=pipelinecard.event_labeler,
    )
