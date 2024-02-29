from collections.abc import Callable, Sequence

from ..utils.list import empty_if_none, flatten
from .classes import (
    Composite,
    Optimizer,
    Pipeline,
    Pipelines,
    Sampler,
    TrainedPipeline,
    TrainedPipelines,
    Transformation,
)


def traverse_apply(pipeline: Pipeline, apply_func: Callable) -> Pipeline:
    def _traverse_apply(pipeline):
        if isinstance(pipeline, list | tuple):
            return [_traverse_apply(t) for t in pipeline]
        return apply_func(pipeline, _traverse_apply)

    return _traverse_apply(pipeline)


def traverse(
    pipeline: Pipeline | list[Pipeline],
):
    """
    Iterates over a pipeline and yields each transformation.
    CAUTION: It does not "unroll" Optimizer's candidates.
    """
    if isinstance(pipeline, list | tuple):
        for i in pipeline:
            yield from traverse(i)
    elif isinstance(pipeline, Composite):
        yield pipeline
        items = pipeline.get_children_primary(only_traversal=True) + empty_if_none(
            pipeline.get_children_secondary()
        )
        for i in items:
            yield from traverse(i)
    elif isinstance(pipeline, Sampler):
        yield pipeline
        yield from traverse(pipeline.get_children_primary(only_traversal=True))
    elif isinstance(pipeline, Optimizer):
        yield pipeline
        if pipeline.get_optimized_pipeline() is not None:
            yield from traverse(pipeline.get_optimized_pipeline())
        else:
            yield from traverse(pipeline.get_candidates(only_traversal=True)[0])
    else:
        yield pipeline


def get_flat_list_of_transformations(
    transformations: Pipelines,
) -> Sequence[Transformation]:
    return flatten(
        [t for t in traverse(transformations) if isinstance(t, Transformation)]
    )


def get_concatenated_names(transformations: Pipelines | Pipeline) -> str:
    return "-".join(
        [
            transformation.name if hasattr(transformation, "name") else ""
            for transformation in get_flat_list_of_transformations(transformations)
        ]
    )


def get_maximum_memory_size(pipeline: Pipeline) -> int:
    memory_sizes = [
        t.properties.memory_size
        for t in get_flat_list_of_transformations(pipeline)
        if t.properties.memory_size is not None
    ]
    if len(memory_sizes) == 0:
        return 0
    return max(memory_sizes)


def get_last_trained_pipeline(trained_pipelines: TrainedPipelines) -> TrainedPipeline:
    return [series.iloc[-1] for series in trained_pipelines]
