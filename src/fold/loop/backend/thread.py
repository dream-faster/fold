# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, List, Optional

import pandas as pd
from tqdm.contrib.concurrent import thread_map

from ...base import Artifact, Composite, Transformations, X
from ...splitters import Fold
from ..types import Backend, Stage


def train_transformations(
    func: Callable,
    transformations: Transformations,
    X: pd.DataFrame,
    y: pd.Series,
    artifact: Artifact,
    splits: List[Fold],
    never_update: bool,
    backend: Backend,
    silent: bool,
):
    return thread_map(
        lambda split: func(
            X, y, artifact, transformations, split, never_update, backend
        ),
        splits,
        disable=silent,
    )


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
