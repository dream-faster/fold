# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional, Union

import pandas as pd

from ..base import Artifact, Composite, Pipelines, Transformation, Tunable
from ..utils.list import wrap_in_list
from .utils import _check_for_duplicate_names


class SelectBest(Composite, Tunable):
    selected_: Optional[str] = None

    def __init__(
        self,
        choose_from: Pipelines,
        name: Optional[str] = None,
    ) -> None:
        self.choose_from = choose_from
        _check_for_duplicate_names(self.choose_from)
        self.name = name or "SelectBest"
        self.properties = Composite.Properties()
        self.metadata = None

    @classmethod
    def from_cloned_instance(
        cls,
        choose_from: Pipelines,
        selected_: Optional[str],
        name: Optional[str],
        metadata: Optional[Composite.Metadata],
        id: str,
    ) -> SelectBest:
        instance = cls(choose_from)
        instance.selected_ = selected_
        instance.name = name
        instance.metadata = metadata
        instance.id = id
        return instance

    def postprocess_result_primary(
        self,
        results: List[pd.DataFrame],
        y: Optional[pd.Series],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        assert self.selected_ is not None, ValueError(
            "SelectBest only works within an `Optimize` class."
        )
        return results[0]

    def get_children_primary(self) -> Pipelines:
        selected = get_candidate_by_name(self.choose_from, self.selected_)
        if selected is None:
            return self.choose_from
        else:
            return wrap_in_list(selected)

    def clone(self, clone_children: Callable) -> SelectBest:
        return SelectBest.from_cloned_instance(
            choose_from=clone_children(self.choose_from),
            selected_=self.selected_,
            name=self.name,
            metadata=self.metadata,
            id=self.id,
        )

    def get_params(self) -> dict:
        return dict(
            selected_=self.selected_,
            name=self.name,
        )

    def get_params_to_try(self) -> Optional[dict]:
        return {"selected_": [i.name for i in self.choose_from]}

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        assert clone_children is not None
        # This is a bit hacky, we "peak" into the new params, and check if selected_ has been set,
        # if yes, we already only clone the selected candidate, not all, to enable conditional hyperparameter optimization.
        # This is our way of filtering out objects that should not be tuned during this round.
        if "selected_" in parameters:
            children = (
                self.choose_from
                if parameters["selected_"] is None
                else [get_candidate_by_name(self.choose_from, parameters["selected_"])]
            )
        else:
            children = self.choose_from
        return SelectBest.from_cloned_instance(
            choose_from=clone_children(children),
            selected_=parameters["selected_"],
            name=self.name,
            metadata=self.metadata,
            id=self.id,
        )


def get_candidate_by_name(
    candidates: List[Union[Transformation, Composite]], name: str
) -> Optional[Union[Transformation, Composite]]:
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    return None
