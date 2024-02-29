# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from enum import Enum

from finml_utils.enums import ParsableEnum


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


class Stage(Enum):
    inital_fit = "inital_fit"
    infer = "infer"

    def is_fit(self) -> bool:
        return self in [Stage.inital_fit]
