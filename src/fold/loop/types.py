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

    @staticmethod
    def from_str(value: Union[str, TrainMethod]) -> TrainMethod:
        if isinstance(value, TrainMethod):
            return value
        for strategy in TrainMethod:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown TrainMethod: {value}")
