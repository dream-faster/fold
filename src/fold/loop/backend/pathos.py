# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, List, Optional

import pandas as pd
from p_tqdm import p_map

from ...base import Artifact, Composite, Pipeline, TrainedPipeline, X
from ...splitters import Fold
from ..types import Backend, Stage


def train_pipeline(
    self,
    func: Callable,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    artifact: Artifact,
    splits: List[Fold],
    never_update: bool,
    backend: Backend,
    silent: bool,
    disable_memory: bool,
):
    return p_map(
        lambda split: func(
            X, y, artifact, pipeline, split, never_update, backend, disable_memory
        ),
        splits,
        disable=silent,
    )


def backtest_pipeline(
    self,
    func: Callable,
    pipeline: TrainedPipeline,
    splits: List[Fold],
    X: pd.DataFrame,
    y: pd.Series,
    artifact: Artifact,
    backend: Backend,
    mutate: bool,
    silent: bool,
    disable_memory: bool,
):
    return p_map(
        lambda split: func(
            pipeline, split, X, y, artifact, backend, mutate, disable_memory
        ),
        splits,
        disable=silent,
    )


def process_child_transformations(
    self,
    func: Callable,
    list_of_child_transformations_with_index: List,
    composite: Composite,
    X: X,
    y: Optional[pd.Series],
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: Optional[List[pd.DataFrame]],
    disable_memory: bool,
):
    list_of_child_transformations_with_index = [
        {"index": index, "child_transformation": child_transformation}
        for index, child_transformation in list_of_child_transformations_with_index
    ]
    return p_map(
        lambda obj: func(
            composite,
            obj["index"],
            obj["child_transformation"],
            X,
            y,
            artifacts,
            stage,
            backend,
            results_primary,
            disable_memory,
        ),
        list_of_child_transformations_with_index,
        disable=True,
    )


class PathosBackend(Backend):
    name = "pathos"
    process_child_transformations = process_child_transformations
    train_pipeline = train_pipeline
    backtest_pipeline = backtest_pipeline
