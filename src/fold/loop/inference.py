from typing import Optional

import pandas as pd

from ..base import Artifact, DeployablePipeline, OutOfSamplePredictions
from .backend import get_backend
from .common import recursively_transform
from .types import BackendType, Stage


def infer(
    pipeline: DeployablePipeline,
    X: Optional[pd.DataFrame],
    disable_memory: bool = False,
) -> OutOfSamplePredictions:
    """
    Run inference on a Pipeline and given data.
    Does not mutate or change the transformations in any way.
    A follow-up call to `update` is required to update the pipeline, when the ground truth is available.
    If X is None it will predict one step ahead.
    """
    if X is None:
        X = pd.DataFrame([0], columns=[0])
    else:
        assert type(X) is pd.DataFrame, "X must be a pandas DataFrame."

    _, results, _ = recursively_transform(
        X,
        None,
        Artifact.empty(X.index),
        pipeline,
        stage=Stage.infer,
        backend=get_backend(BackendType.no),
        disable_memory=disable_memory,
    )
    return results
