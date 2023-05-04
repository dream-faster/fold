# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from copy import deepcopy
from typing import Callable, List

from ..base import (
    Composite,
    Optimizer,
    Pipeline,
    Pipelines,
    Transformation,
    Transformations,
)
from ..utils.list import flatten, wrap_in_list


def traverse_pipeline(pipeline: Pipeline, apply_func: Callable) -> Pipeline:
    def _traverse(pipeline):
        if isinstance(pipeline, List):
            return [_traverse(t) for t in pipeline]
        elif isinstance(pipeline, Composite):
            composite = pipeline
            return composite.clone(clone_child_transformations=_traverse)
        elif isinstance(pipeline, Optimizer):
            raise ValueError("Optimizer is not supported within an Optimizer.")
        elif isinstance(pipeline, Transformation):
            return apply_func(deepcopy(pipeline))

    return _traverse(pipeline)


def get_flat_list_of_transformations(
    transformations: Pipelines,
) -> List[Transformations]:
    def get_all_transformations(transformations: Pipelines) -> Transformations:
        if isinstance(transformations, List):
            return [get_all_transformations(t) for t in transformations]
        elif isinstance(transformations, Composite):
            secondary_transformations = (
                []
                if (transformations.get_child_transformations_secondary()) is None
                else transformations.get_child_transformations_secondary()
            )
            return [
                get_all_transformations(child_t)
                for child_t in transformations.get_child_transformations_primary()
            ] + [
                get_all_transformations(child_t)
                for child_t in secondary_transformations
            ]
        else:
            return transformations

    return flatten(wrap_in_list(get_all_transformations(transformations)))


def get_concatenated_names(transformations: Pipelines) -> str:
    return "-".join(
        [
            transformation.name if hasattr(transformation, "name") else ""
            for transformation in get_flat_list_of_transformations(transformations)
        ]
    )
