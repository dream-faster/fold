from __future__ import annotations

from enum import Enum
from typing import Union


class Backend(Enum):
    no = "no"
    ray = "ray"

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
    parallel = "parallel"
    sequential = "sequential"
    parallel_with_search = (
        "parallel_with_search"  # Don't use it just yet, not yet fully documented
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
