# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from collections.abc import Callable

import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

from ...base import Artifact, Backend, Composite, Pipeline, TrainedPipeline, X
from ...splitters import Fold
from ..types import Stage

DEBUG_MULTI_PROCESSING = False


def train_pipeline(
    self,
    func: Callable,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    artifact: Artifact,
    splits: list[Fold],
    backend: Backend,
    project_name: str,
    project_hyperparameters: dict | None,
    preprocessing_max_memory_size: int,
    silent: bool,
):
    if DEBUG_MULTI_PROCESSING:
        from ray.cloudpickle import dumps, loads

        pipeline = loads(dumps(pipeline))
    return [
        func(
            X,
            y,
            artifact,
            pipeline,
            split,
            backend,
            project_name,
            project_hyperparameters,
            preprocessing_max_memory_size,
        )
        for split in tqdm_auto(splits, desc="Training", disable=silent)
    ]


def backtest_pipeline(
    self,
    func: Callable,
    pipeline: TrainedPipeline,
    splits: list[Fold],
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
    list_of_child_transformations_with_index: list,
    composite: Composite,
    X: X,
    y: pd.Series | None,
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: list[pd.DataFrame] | None,
    tqdm: tqdm | None = None,
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
