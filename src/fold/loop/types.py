# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from enum import Enum
from typing import Union


class Backend(Enum):
    """
    Parameters
    ----------
    no: string
        Uses sequential processing. This is the default.
    ray: string
        Uses `ray` as a backend. Call `ray.init()` before using this backend.
    process: string
        Uses `multiprocessing` as a backend (via tqdm.contrib.concurrent.process_map).
    thread: string
        Uses `threading` as a backend (via tqdm.contrib.concurrent.thread_map).

    """

    no = "no"
    ray = "ray"
    process = "process"
    thread = "thread"

    @staticmethod
    def from_str(value: Union[str, Backend]) -> Backend:
        if isinstance(value, Backend):
            return value
        for strategy in Backend:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown Backend: {value}")


class TrainMethod(Enum):
    """
    Parameters
    ----------
    parallel: string
        Parallel, independent training of pipelines for each fold.
    sequential: string
        Sequentially train/update pipelines, walking forward in time.

    """

    parallel = "parallel"
    sequential = "sequential"
    parallel_with_search = (  # Don't use it just yet, not yet fully documented
        "parallel_with_search"
    )

    @staticmethod
    def from_str(value: Union[str, TrainMethod]) -> TrainMethod:
        if isinstance(value, TrainMethod):
            return value
        for strategy in TrainMethod:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown TrainMethod: {value}")


class Stage(Enum):
    inital_fit = "inital_fit"
    update = "update"
    update_online_only = "update_online_only"
    infer = "infer"

    def is_fit_or_update(self) -> bool:
        return self in [Stage.inital_fit, Stage.update]
