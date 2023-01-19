import collections
import itertools
from typing import List, TypeVar, Union

T = TypeVar("T")


def wrap_into_list_if_needed(input: Union[T, List[T]]) -> List[T]:
    return input if isinstance(input, List) else [input]


def flatten(input: List[List]) -> List:
    return list(itertools.chain.from_iterable(input))


def keep_only_duplicates(input: List) -> List:
    return [item for item, count in collections.Counter(input).items() if count > 1]


def has_intersection(lhs: List, rhs: List) -> bool:
    return len(set(lhs).intersection(rhs)) > 0
