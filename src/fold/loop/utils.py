# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from copy import deepcopy
from typing import Callable, List

import pandas as pd

from ..base import (
    Block,
    Composite,
    Pipeline,
    TrainedPipelines,
    Transformations,
    get_flat_list_of_transformations,
)


def deepcopy_pipelines(transformation: Transformations) -> Transformations:
    if isinstance(transformation, List):
        return [deepcopy_pipelines(t) for t in transformation]
    elif isinstance(transformation, Composite):
        return transformation.clone(deepcopy_pipelines)
    else:
        return deepcopy(transformation)


def replace_with(transformations: List[Block]) -> Callable:
    transformations = get_flat_list_of_transformations(transformations)
    mapping = {t.id: t for t in transformations}

    def replace(transformation: Transformations) -> Transformations:
        if isinstance(transformation, List):
            return [replace(t) for t in transformation]
        elif isinstance(transformation, Composite):
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
