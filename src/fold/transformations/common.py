from typing import TYPE_CHECKING, List

from ..transformations.base import Transformation, Transformations
from ..utils.list import flatten, wrap_in_list

if TYPE_CHECKING:
    from ..composites.base import Composite


def get_flat_list_of_transformations(
    transformations: Transformations,
) -> List[Transformation]:
    def get_all_transformations(transformations: Transformations) -> Transformations:
        if isinstance(transformations, List):
            return [get_all_transformations(t) for t in transformations]
        elif isinstance(transformations, Composite):
            secondary_transformations = (
                []
                if (transformations.get_child_transformations_secondary()) is None
                else transformations.get_child_transformations_secondary()
            )
            return [
                get_all_transformations(child_t)
                for child_t in transformations.get_child_transformations_primary()
            ] + [
                get_all_transformations(child_t)
                for child_t in secondary_transformations
            ]
        else:
            return transformations

    return flatten(wrap_in_list(get_all_transformations(transformations)))


def get_concatenated_names(transformations: Transformations) -> str:
    return "-".join(
        [
            transformation.name if hasattr(transformation, "name") else ""
            for transformation in get_flat_list_of_transformations(transformations)
        ]
    )
