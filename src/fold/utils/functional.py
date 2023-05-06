from typing import Callable, TypeVar

T = TypeVar("T", object, None)


def apply_if_not_none(value: T, func: Callable) -> T:
    if value is not None:
        return func(value)
    return None
