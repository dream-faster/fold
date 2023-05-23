# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional, Union

import pandas as pd

from ..base import Composite, Pipelines, Transformation, Tunable, get_concatenated_names
from ..utils.list import wrap_in_list


class SelectBest(Composite, Tunable):
    properties = Composite.Properties()
    selected_: Optional[str] = None

    def __init__(
        self,
        choose_from: List[Union[Transformation, Composite]],
        name: Optional[str] = None,
    ) -> None:
        self.choose_from = choose_from
        for i in self.choose_from:
            if isinstance(i, Tunable) and i.get_params_to_try() is not None:
                raise ValueError(
                    "You can not simulatenously select a model and tune its parameters right now."
                )
        names = [i.name for i in self.choose_from]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate names in `choose_from` are not allowed.")
        self.name = name or "SelectBest" + get_concatenated_names(self.choose_from)

    @classmethod
    def from_cloned_instance(
        cls,
        choose_from: List[Union[Transformation, Composite]],
        selected_: Optional[str],
        name: Optional[str],
    ) -> SelectBest:
        instance = cls(choose_from)
        instance.selected_ = selected_
        instance.name = name
        return instance

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        assert self.selected_ is not None, ValueError(
            "SelectBest only works within an `Optimize` class."
        )
        return results[0]

    def get_children_primary(self) -> Pipelines:
        selected = get_candidate_by_name(self.choose_from, self.selected_)
        if selected is None:
            return wrap_in_list(self.choose_from[0])
        else:
            return wrap_in_list(selected)

    def clone(self, clone_children: Callable) -> SelectBest:
        return SelectBest.from_cloned_instance(
            choose_from=clone_children(self.choose_from),
            selected_=self.selected_,
            name=self.name,
        )

    def get_params(self) -> dict:
        return {"selected_": self.selected_, "name": self.name}

    def get_params_to_try(self) -> Optional[dict]:
        return {"selected_": [i.name for i in self.choose_from]}

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        assert clone_children is not None
        return SelectBest.from_cloned_instance(
            choose_from=clone_children(self.choose_from),
            selected_=parameters["selected_"],
            name=self.name,
        )


def get_candidate_by_name(
    candidates: List[Union[Transformation, Composite]], name: str
) -> Optional[Union[Transformation, Composite]]:
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    return None
