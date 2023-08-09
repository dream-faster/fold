# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from copy import deepcopy
from typing import Callable, List, Tuple, TypeVar

import pandas as pd

from ..base import Block, Clonable, Composite, Pipeline, TrainedPipelines
from ..base.utils import get_maximum_memory_size, traverse, traverse_apply
from ..splitters import Fold
from .types import Stage


def deepcopy_pipelines(transformation: Pipeline) -> Pipeline:
    if isinstance(transformation, List) or isinstance(transformation, Tuple):
        return [deepcopy_pipelines(t) for t in transformation]
    elif isinstance(transformation, Clonable):
        return transformation.clone(deepcopy_pipelines)
    else:
        return deepcopy(transformation)


def replace_with(transformations: List[Block]) -> Callable:
    mapping = {t.id: t for t in traverse(transformations)}

    def replace(transformation: Pipeline) -> Pipeline:
        if isinstance(transformation, List) or isinstance(transformation, Tuple):
            return [replace(t) for t in transformation]

        if transformation.id in mapping:
            return mapping[transformation.id]

        if isinstance(transformation, Clonable):
            return transformation.clone(replace)

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


T = TypeVar("T")


def _cut_to_train_window(df: T, fold: Fold, stage: Stage) -> T:
    window_start = (
        fold.update_window_start if stage == Stage.update else fold.train_window_start
    )
    window_end = (
        fold.update_window_end if stage == Stage.update else fold.train_window_end
    )
    return df.iloc[window_start:window_end]  # type: ignore


def _cut_to_backtesting_window(df: T, fold: Fold, pipeline: Pipeline) -> T:
    overlap = get_maximum_memory_size(pipeline)
    test_window_start = max(fold.test_window_start - overlap, 0)
    return df.iloc[test_window_start : fold.test_window_end]  # type: ignore
