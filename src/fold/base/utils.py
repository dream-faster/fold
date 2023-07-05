from typing import Callable, List, Tuple

from ..utils.list import empty_if_none, flatten
from .classes import (
    Composite,
    Optimizer,
    Pipeline,
    Pipelines,
    Sampler,
    Transformation,
    Transformations,
)


def traverse_apply(pipeline: Pipeline, apply_func: Callable) -> Pipeline:
    def _traverse_apply(pipeline):
        if isinstance(pipeline, List) or isinstance(pipeline, Tuple):
            return [_traverse_apply(t) for t in pipeline]
        elif isinstance(pipeline, Optimizer):
            raise ValueError("Optimizer is not supported within an Optimizer.")
        elif (
            isinstance(pipeline, Transformation)
            or isinstance(pipeline, Composite)
            or isinstance(pipeline, Sampler)
        ):
            return apply_func(pipeline, _traverse_apply)

    return _traverse_apply(pipeline)


def traverse(
    pipeline: Pipeline,
):
    """
    Iterates over a pipeline and yields each transformation.
    CAUTION: It does not "unroll" Optimizer's candidates.
    """
    if isinstance(pipeline, List) or isinstance(pipeline, Tuple):
        for i in pipeline:
            yield from traverse(i)
    elif isinstance(pipeline, Composite):
        yield pipeline
        items = pipeline.get_children_primary() + empty_if_none(
            pipeline.get_children_secondary()
        )
        for i in items:
            yield from traverse(i)
    elif isinstance(pipeline, Sampler):
        yield pipeline
        yield from traverse(pipeline.get_children_primary())
    elif isinstance(pipeline, Optimizer):
        raise ValueError("Optimizer is not supported within an Optimizer.")
    else:
        yield pipeline


def get_flat_list_of_transformations(
    transformations: Pipelines,
) -> List[Transformations]:
    return flatten(
        [t for t in traverse(transformations) if isinstance(t, Transformation)]
    )


def get_concatenated_names(transformations: Pipelines) -> str:
    return "-".join(
        [
            transformation.name if hasattr(transformation, "name") else ""
            for transformation in get_flat_list_of_transformations(transformations)
        ]
    )
