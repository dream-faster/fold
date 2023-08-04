# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from ..utils.enums import ParsableEnum


@dataclass
class Backend:
    name: str
    process_child_transformations: Callable
    train_pipeline: Callable
    backtest_pipeline: Callable

    def __init__(self):  # to be able to initalize it with the default parameters
        pass


class BackendType(ParsableEnum):
    """
    Parameters
    ----------
    no: string
        Uses sequential processing. This is the default.
    ray: string
        Uses `ray` as a backend. Call `ray.init()` before using this backend.
    pathos: string
        Uses `pathos.multiprocessing` as a backend (via [p_tqdm](https://github.com/swansonk14/p_tqdm)).
    thread: string
        Uses `threading` as a backend (via tqdm.contrib.concurrent.thread_map).

    """

    no = "no"
    ray = "ray"
    pathos = "pathos"
    thread = "thread"
    joblib = "joblib"


class TrainMethod(ParsableEnum):
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


class Stage(Enum):
    inital_fit = "inital_fit"
    update = "update"
    update_online_only = "update_online_only"
    infer = "infer"

    def is_fit_or_update(self) -> bool:
        return self in [Stage.inital_fit, Stage.update]
