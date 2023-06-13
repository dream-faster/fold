# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, List, Optional

import pandas as pd
import ray

from ...base import Artifact, Composite, Transformations, X
from ...splitters import Fold
from ..types import Backend, Stage


def train_transformations(
    func: Callable,
    transformations: Transformations,
    X: X,
    y: pd.Series,
    artifact: Artifact,
    splits: List[Fold],
    never_update: bool,
    backend: Backend,
    silent: bool,
):
    func = ray.remote(func)
    X = ray.put(X)
    y = ray.put(y)
    return ray.get(
        [
            func.remote(
                X,
                y,
                artifact,
                transformations,
                split,
                never_update,
                backend,
            )
            for split in splits
        ]
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


# def process_child_transformations(
#     func: Callable,
#     list_of_child_transformations_with_index: List,
#     composite: Composite,
#     X: X,
#     y: Optional[pd.Series],
#     sample_weights: Optional[pd.Series],
#     artifacts: Artifact,
#     stage: Stage,
#     backend: Backend,
#     results_primary: Optional[List[pd.DataFrame]],
# ):
#     func = ray.remote(func)
#     X = ray.put(X)
#     y = ray.put(y)
#     return ray.get(
#         [
#             func.remote(
#                 composite,
#                 index,
#                 child_transformation,
#                 X,
#                 y,
#                 sample_weights,
#                 artifacts,
#                 stage,
#                 backend,
#                 results_primary,
#             )
#             for index, child_transformation in list_of_child_transformations_with_index
#         ]
#     )
