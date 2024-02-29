# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from ...base import Artifact, Backend, Composite, Pipeline, TrainedPipeline, X
from ...splitters import Fold
from ..types import Stage


def train_pipeline(
    self: JoblibBackend,
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
    return Parallel(
        n_jobs=self.limit_threads, prefer="threads" if self.prefer_threads else None
    )(
        [
            delayed(func)(
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
            for split in splits
        ]
    )


def backtest_pipeline(
    self: JoblibBackend,
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
    return Parallel(
        n_jobs=self.limit_threads, prefer="threads" if self.prefer_threads else None
    )(
        [
            delayed(func)(pipeline, split, X, y, artifact, backend, mutate)
            for split in splits
        ]
    )


def process_child_transformations(
    self: JoblibBackend,
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
    list_of_child_transformations_with_index = list(
        list_of_child_transformations_with_index
    )
    # if len(list_of_child_transformations_with_index) == 1:
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

    # with parallel_config(backend="threading"):
    #     return Parallel(
    #         n_jobs=self.limit_threads, prefer="threads" if self.prefer_threads else None
    #     )(
    #         [
    #             delayed(func)(
    #                 composite,
    #                 index,
    #                 child_transformation,
    #                 X,
    #                 y,
    #                 artifacts,
    #                 stage,
    #                 backend,
    #                 results_primary,
    #                 None,
    #             )
    #             for index, child_transformation in list_of_child_transformations_with_index
    #         ]
    #     )


class JoblibBackend(Backend):
    name = "joblib"
    process_child_transformations = process_child_transformations
    train_pipeline = train_pipeline
    backtest_pipeline = backtest_pipeline

    def __init__(self, limit_threads: int | None = None, prefer_threads: bool = True):
        self.limit_threads = limit_threads
        self.prefer_threads = prefer_threads
