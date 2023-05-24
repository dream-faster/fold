from typing import List

from ..base import Pipeline, Tunable, traverse


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
