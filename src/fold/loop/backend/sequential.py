# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, List, Optional

import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

from ...base import Artifact, Composite, Pipeline, TrainedPipeline, X
from ...splitters import Fold
from ..types import Backend, Stage

DEBUG_MULTI_PROCESSING = False


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
    project_name: str,
    silent: bool,
):
    if DEBUG_MULTI_PROCESSING:
        from ray.cloudpickle import dumps, loads

        pipeline = loads(dumps(pipeline))
    return [
        func(X, y, artifact, pipeline, split, never_update, backend, project_name)
        for split in tqdm_auto(splits, desc="Training", disable=silent)
    ]


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
):
    if DEBUG_MULTI_PROCESSING:
        from ray.cloudpickle import dumps, loads

        pipeline = loads(dumps(pipeline))
    return [
        func(pipeline, split, X, y, artifact, backend, mutate)
        for split in tqdm(splits, desc="Backtesting", disable=silent)
    ]


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
    tqdm: Optional[tqdm] = None,
):
    if DEBUG_MULTI_PROCESSING:
        from ray.cloudpickle import dumps, loads

        list_of_child_transformations_with_index = loads(
            dumps(list_of_child_transformations_with_index)
        )
    return [
        func(
            composite,
            index,
            child_transformation,
            X,
            y,
            artifacts,
            stage,
            backend,
            results_primary,
            tqdm,
        )
        for index, child_transformation in list_of_child_transformations_with_index
    ]


class NoBackend(Backend):
    name = "no"
    process_child_transformations = process_child_transformations
    train_pipeline = train_pipeline
    backtest_pipeline = backtest_pipeline
