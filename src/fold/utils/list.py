import collections
from collections.abc import Iterable
from typing import List, Tuple, TypeVar, Union

from iteration_utilities import unique_everseen

T = TypeVar("T")


def wrap_in_list(input: Union[T, List[T]]) -> List[T]:
    return input if isinstance(input, List) else [input]


def flatten(input: Union[List[List], List]) -> Iterable:
    for x in input:
        if isinstance(x, list):
            yield from flatten(x)
        else:
            yield x


def keep_only_duplicates(input: List) -> List:
    return [item for item, count in collections.Counter(input).items() if count > 1]


def has_intersection(lhs: List, rhs: List) -> bool:
    return len(set(lhs).intersection(rhs)) > 0


def unique(input: List) -> List:
    return unique_everseen(input)


def swap_tuples(input: List[Tuple]) -> List[Tuple]:
    return [(b, a) for a, b in input]
