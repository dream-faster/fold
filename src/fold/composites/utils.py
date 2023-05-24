from copy import copy
from typing import Callable, List, Union

from ..base import Composite, Pipeline, Transformation, Tunable, traverse
from ..transformations.dev import Identity


def _process_params(params_to_try: dict) -> dict:
    if "passthrough" in params_to_try:
        params_to_try["passthrough"] = [True, False]
    return params_to_try


def _get_tunables_with_params_to_try(pipeline: Pipeline) -> List[Tunable]:
    return [
        i
        for i in traverse(pipeline)
        if isinstance(i, Tunable) and i.get_params_to_try() is not None
    ]


def _apply_params(params: dict) -> Callable:
    def __apply_params_to_transformation(
        item: Union[Composite, Transformation], clone_children: Callable
    ) -> Union[Composite, Transformation]:
        if not isinstance(item, Tunable):
            if isinstance(item, Composite):
                return item.clone(clone_children)
            else:
                return item
        selected_params = params.get(item.name, {})
        if "passthrough" in selected_params and selected_params["passthrough"] is True:
            return Identity()  # type: ignore
        return item.clone_with_params(
            parameters={**item.get_params(), **_clean_params(selected_params)},
            clone_children=clone_children,
        )  # type: ignore

    return __apply_params_to_transformation


def _clean_params(params_to_try: dict) -> dict:
    if "passthrough" in params_to_try:
        params_to_try = copy(params_to_try)
        del params_to_try["passthrough"]
    return params_to_try


def _check_for_duplicate_names(pipeline: Pipeline):
    names = [i.name for i in traverse(pipeline)]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate names in pipeline are not allowed.")
