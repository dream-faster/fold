# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from dataclasses import dataclass
from typing import Callable

from fold.loop.types import Backend


@dataclass
class BackendDependentFunctions:
    process_child_transformations: Callable
    train_pipeline: Callable
    backtest_pipeline: Callable


def get_backend_dependent_functions(backend: Backend) -> BackendDependentFunctions:
    if backend == Backend.ray:
        from .ray import (
            backtest_pipeline,
            process_child_transformations,
            train_pipeline,
        )
    elif backend == Backend.no:
        from .sequential import (
            backtest_pipeline,
            process_child_transformations,
            train_pipeline,
        )
    elif backend == Backend.pathos:
        from .pathos import (
            backtest_pipeline,
            process_child_transformations,
            train_pipeline,
        )
    elif backend == Backend.thread:
        from .thread import (
            backtest_pipeline,
            process_child_transformations,
            train_pipeline,
        )
    else:
        raise ValueError(f"Backend {backend} not supported.")

    return BackendDependentFunctions(
        process_child_transformations, train_pipeline, backtest_pipeline
    )
