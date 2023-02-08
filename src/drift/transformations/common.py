from typing import List

from ..transformations.base import Composite, Transformations
from ..utils.list import flatten


def get_flat_list_of_transformations(
    transformations: Transformations,
) -> List[Transformations]:
    def get_all_transformations(transformations: Transformations) -> Transformations:
        if isinstance(transformations, List):
            return [get_all_transformations(t) for t in transformations]
        elif isinstance(transformations, Composite):
            return [
                get_all_transformations(child_t)
                for child_t in transformations.get_child_transformations()
            ]
        else:
            return transformations

    return flatten(get_all_transformations(transformations))
