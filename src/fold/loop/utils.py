# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TypeVar

import pandas as pd

from ..base import Block, BlockMetadata, Clonable, Pipeline, TrainedPipelines
from ..base.utils import get_maximum_memory_size, traverse, traverse_apply
from ..splitters import Bounds, Fold
from .types import Stage


def deepcopy_pipelines(transformation: Pipeline) -> Pipeline:
    if isinstance(transformation, list | tuple):
        return [deepcopy_pipelines(t) for t in transformation]
    if isinstance(transformation, Clonable):
        return transformation.clone(deepcopy_pipelines)
    return deepcopy(transformation)


def replace_with(transformations: Sequence[Block]) -> Callable:
    mapping = {t.id: t for t in traverse(transformations)}

    def replace(transformation: Pipeline) -> Pipeline:
        if isinstance(transformation, list | tuple):
            return [replace(t) for t in transformation]

        if transformation.id in mapping:
            return mapping[transformation.id]

        if isinstance(transformation, Clonable):
            return transformation.clone(replace)

        return transformation

    return replace


def _extract_trained_pipelines(
    processed_idx: list[int], processed_pipelines: list[Pipeline]
) -> TrainedPipelines:
    return [
        pd.Series(
            transformation_over_time,
            index=processed_idx,
            name=transformation_over_time[0].name,
        )
        for transformation_over_time in zip(*processed_pipelines, strict=True)
    ]


def _set_metadata(
    pipeline: Pipeline,
    metadata: BlockMetadata,
) -> Pipeline:
    def set_(block: Block, clone_children: Callable) -> Block:
        if isinstance(block, Clonable):
            block = block.clone(clone_children)
        block.metadata = metadata
        return block

    return traverse_apply(pipeline, set_)


T = TypeVar("T")


def _cut_to_train_window(df: T, fold: Fold, stage: Stage) -> T:
    return df.iloc[fold.train_indices()]  # type: ignore


def _cut_to_backtesting_window(df: T, fold: Fold, pipeline: Pipeline) -> T:
    overlap = get_maximum_memory_size(pipeline)

    def enlargen_test_if_needed(bounds: Bounds) -> Bounds:
        if bounds.start - overlap >= 0:
            return Bounds(bounds.start - overlap, bounds.end)
        return Bounds(0, bounds.end)

    if overlap > 0:
        fold = Fold(
            index=fold.index,
            train_bounds=fold.train_bounds,
            test_bounds=[enlargen_test_if_needed(b) for b in fold.test_bounds],
        )

    return df.iloc[fold.test_indices()]  # type: ignore
