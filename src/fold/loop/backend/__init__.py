# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from dataclasses import dataclass
from typing import Callable

from fold.loop.types import Backend


@dataclass
class BackendDependentFunctions:
    process_primary_child_transformations: Callable
    process_secondary_child_transformations: Callable
    train_transformations: Callable


def get_backend_dependent_functions(backend: Backend) -> BackendDependentFunctions:
    if backend == Backend.ray:
        from .ray import (
            process_primary_child_transformations,
            process_secondary_child_transformations,
            train_transformations,
        )

        return BackendDependentFunctions(
            process_primary_child_transformations,
            process_secondary_child_transformations,
            train_transformations,
        )
    elif backend == Backend.no:
        from .sequential import (
            process_primary_child_transformations,
            process_secondary_child_transformations,
            train_transformations,
        )

        return BackendDependentFunctions(
            process_primary_child_transformations,
            process_secondary_child_transformations,
            train_transformations,
        )
