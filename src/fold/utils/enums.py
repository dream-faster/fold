from __future__ import annotations

from enum import Enum
from typing import Union


class ParsableEnum(Enum):
    @classmethod
    def from_str(cls, value: Union[str, ParsableEnum]) -> ParsableEnum:
        if isinstance(value, cls):
            return value
        for item in cls:
            if item.value == value:
                return item
        else:
            raise ValueError(f"Unknown {cls.__name__}: {value}")
