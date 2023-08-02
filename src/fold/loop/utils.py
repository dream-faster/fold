# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from copy import deepcopy
from typing import Callable, List, Tuple

import pandas as pd

from fold.base.utils import traverse_apply

from ..base import (
    Block,
    Clonable,
    Composite,
    Pipeline,
    TrainedPipelines,
    get_flat_list_of_transformations,
)


def deepcopy_pipelines(transformation: Pipeline) -> Pipeline:
    if isinstance(transformation, List) or isinstance(transformation, Tuple):
        return [deepcopy_pipelines(t) for t in transformation]
    elif isinstance(transformation, Clonable):
        return transformation.clone(deepcopy_pipelines)
    else:
        return deepcopy(transformation)


def replace_with(transformations: List[Block]) -> Callable:
    transformations = get_flat_list_of_transformations(transformations)
    mapping = {t.id: t for t in transformations}

    def replace(transformation: Pipeline) -> Pipeline:
        if isinstance(transformation, List) or isinstance(transformation, Tuple):
            return [replace(t) for t in transformation]
        elif isinstance(transformation, Clonable):
            return transformation.clone(replace)
        else:
            if transformation.id in mapping:
                return mapping[transformation.id]
            else:
                return transformation

    return replace


def _extract_trained_pipelines(
    processed_idx: List[int], processed_pipelines: List[Pipeline]
) -> TrainedPipelines:
    return [
        pd.Series(
            transformation_over_time,
            index=processed_idx,
            name=transformation_over_time[0].name,
        )
        for transformation_over_time in zip(*processed_pipelines)
    ]


def _set_metadata(
    pipeline: Pipeline,
    metadata: Composite.Metadata,
) -> Pipeline:
    def set_(block: Block, clone_children: Callable) -> Block:
        if isinstance(block, Clonable):
            block = block.clone(clone_children)
            if isinstance(block, Composite):
                block.metadata = metadata
        return block

    return traverse_apply(pipeline, set_)
