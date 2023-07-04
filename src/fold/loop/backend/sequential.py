# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from ...base import Artifact, Composite, Pipeline, TrainedPipeline, X
from ...splitters import Fold
from ..types import Backend, Stage


def train_pipeline(
    func: Callable,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    artifact: Artifact,
    splits: List[Fold],
    never_update: bool,
    backend: Backend,
    silent: bool,
):
    return [
        func(X, y, artifact, pipeline, split, never_update, backend)
        for split in tqdm(splits, desc="Training", disable=silent)
    ]


def backtest_pipeline(
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
    return [
        func(pipeline, split, X, y, artifact, backend, mutate)
        for split in tqdm(splits, desc="Backtesting", disable=silent)
    ]


def process_child_transformations(
    func: Callable,
    list_of_child_transformations_with_index: List,
    composite: Composite,
    X: X,
    y: Optional[pd.Series],
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: Optional[List[pd.DataFrame]],
):
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
        )
        for index, child_transformation in list_of_child_transformations_with_index
    ]
