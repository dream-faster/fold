from typing import Callable, List

from .base import Composite, Optimizer, Pipeline, Transformation
from .utils.list import empty_if_none


def traverse_apply(pipeline: Pipeline, apply_func: Callable) -> Pipeline:
    def _traverse_apply(pipeline):
        if isinstance(pipeline, List):
            return [_traverse_apply(t) for t in pipeline]
        elif isinstance(pipeline, Composite):
            composite = pipeline
            return composite.clone(clone_child_transformations=_traverse_apply)
        elif isinstance(pipeline, Optimizer):
            raise ValueError("Optimizer is not supported within an Optimizer.")
        elif isinstance(pipeline, Transformation):
            return apply_func(pipeline)

    return _traverse_apply(pipeline)


def traverse(
    pipeline: Pipeline,
):
    """
    Iterates over a pipeline and yields each transformation.
    CAUTION: It does not "unroll" the candidate.
    """
    if isinstance(pipeline, List):
        for i in pipeline:
            yield from traverse(i)
    elif isinstance(pipeline, Composite):
        items = pipeline.get_child_transformations_primary() + empty_if_none(
            pipeline.get_child_transformations_secondary()
        )
        for i in items:
            yield from traverse(i)
    elif isinstance(pipeline, Optimizer):
        raise ValueError("Optimizer is not supported within an Optimizer.")
    else:
        yield pipeline
