from typing import Optional

import pandas as pd

from fold.base.classes import DeployablePipelineCard

from ..base import (
    Artifact,
    DeployablePipeline,
    OutOfSamplePredictions,
    get_maximum_memory_size,
)
from .backend import get_backend
from .checks import check_types_X
from .common import recursively_transform
from .types import BackendType, Stage


def infer(
    pipelinecard: DeployablePipelineCard,
    X: Optional[pd.DataFrame],
) -> OutOfSamplePredictions:
    """
    Run inference on a Pipeline and given data.
    Does not mutate or change the transformations in any way.
    A follow-up call to `update` is required to update the pipeline, when the ground truth is available.
    If X is None it will predict one step ahead.
    """
    X = check_types_X(X)
    maximum_memory_size = max(
        get_maximum_memory_size(pipelinecard.preprocessing),
        get_maximum_memory_size(pipelinecard.pipeline),
    )
    assert len(X) > maximum_memory_size, (
        f"X must be larger than {maximum_memory_size} rows, "
        f"but only {len(X)} rows were given."
    )

    if pipelinecard.preprocessing is None:
        _, X, _ = recursively_transform(
            X,
            None,
            Artifact.empty(X.index),
            pipelinecard.preprocessing,
            stage=Stage.infer,
            backend=get_backend(BackendType.no),
        )
    _, results, _ = recursively_transform(
        preprocessing_results,
        None,
        Artifact.empty(X.index),
        pipelinecard.pipeline,
        stage=Stage.infer,
        backend=get_backend(BackendType.no),
    )
    return results
