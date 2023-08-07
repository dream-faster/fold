import pandas as pd

from fold.base.classes import TrainedPipelineCard

from ..base import Artifact, OutOfSamplePredictions, get_maximum_memory_size
from .backend import get_backend
from .common import recursively_transform
from .types import BackendType, Stage


def infer(
    pipelinecard: TrainedPipelineCard,
    X: pd.DataFrame,
) -> OutOfSamplePredictions:
    """
    Run inference on a Pipeline and given data.
    Does not mutate or change the transformations in any way.
    A follow-up call to `update` is required to update the pipeline, when the ground truth is available.
    If X is None it will predict one step ahead.
    """
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame."
    maximum_memory_size = max(
        get_maximum_memory_size(pipelinecard.preprocessing),
        get_maximum_memory_size(pipelinecard.pipeline),
    )
    assert len(X) > maximum_memory_size, (
        f"X must be larger than {maximum_memory_size} rows, "
        f"but only {len(X)} rows were given."
    )

    if pipelinecard.preprocessing is not None:
        _, X, _ = recursively_transform(
            X,
            None,
            Artifact.dummy_events(X.index),
            pipelinecard.preprocessing,
            stage=Stage.infer,
            backend=get_backend(BackendType.no),
        )

    last_pipeline = [series.iloc[-1] for series in pipelinecard.pipeline[-1:]]

    _, results, _ = recursively_transform(
        X,
        None,
        Artifact.dummy_events(X.index),
        last_pipeline,
        stage=Stage.infer,
        backend=get_backend(BackendType.no),
    )
    return results
