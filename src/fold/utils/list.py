# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


import collections
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple, TypeVar, Union

from iteration_utilities import unique_everseen

T = TypeVar("T")


def wrap_in_list(input: Union[T, List[T]]) -> List[T]:
    return input if isinstance(input, List) else [input]


def transform_range_to_list(input: Union[range, List[T]]) -> List[T]:
    return list(input) if isinstance(input, range) else input


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


def filter_none(input: List) -> List:
    return [x for x in input if x is not None]


def empty_if_none(input: Union[List, None]) -> List:
    return [] if input is None else input


def to_hierachical_dict(flat_dict: dict, seperator: str = "¦") -> dict:
    hierarchy = {}
    for key, value in flat_dict.items():
        name_and_param = key.split(seperator)
        unraveled_obj_key = name_and_param[0]
        unraveled_param_key = name_and_param[1]
        if unraveled_obj_key not in hierarchy:
            hierarchy[unraveled_obj_key] = {}
        hierarchy[unraveled_obj_key][unraveled_param_key] = value
    return hierarchy


def to_hierachical_dict_arbitrary_depth(flat_dict: dict, separator: str = "¦") -> dict:
    hierarchy = {}
    for key, value in flat_dict.items():
        current_dict = hierarchy
        parts = key.split(separator)
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        current_dict[parts[-1]] = value
    return hierarchy


def ensure_dict(dictionary: Optional[Dict]) -> dict:
    return {} if dictionary is None else dictionary


def unpack_list_of_tuples(input: List[Tuple]):
    if len(input) == 1:
        return [[item] for item in input[0]]
    return zip(*input)
