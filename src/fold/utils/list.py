import collections
from collections.abc import Iterable
from typing import List, Tuple, TypeVar, Union

from iteration_utilities import unique_everseen

T = TypeVar("T")


def wrap_in_list(input: Union[T, List[T]]) -> List[T]:
    return input if isinstance(input, List) else [input]


def wrap_in_double_list_if_needed(
    input: Union[T, List[T]]
) -> Union[List[List[T]], List[T]]:
    """
    If input is a single item, wrap it in a list.
    If input is a single list, wrap it in another list.
    If input is a list of lists, return it as is.
    """
    if not isinstance(input, List):
        return [input]
    if isinstance(input[0], List):
        return input
    else:
        return [input]


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
