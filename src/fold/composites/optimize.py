# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd
from sklearn.model_selection import ParameterGrid

from fold.utils.list import wrap_in_list

from ..base import Composite, Pipelines, Tuneable
from .common import get_concatenated_names


class OptimizeGridSearch(Composite):
    """
    Don't use this just yet. Coming later.
    """

    properties = Composite.Properties(primary_requires_predictions=True)
    selected_params: Optional[dict] = None

    def __init__(
        self,
        model: Tuneable,
        param_grid: dict,
        scorer: Callable,
        is_scorer_loss: bool = True,
    ) -> None:
        self.model = wrap_in_list(model)
        self.param_grid = param_grid
        self.name = "OptimizeGridSearch-" + get_concatenated_names(self.model)
        self.scorer = scorer
        self.is_scorer_loss = is_scorer_loss

    @classmethod
    def from_cloned_instance(
        cls,
        model: Tuneable,
        param_grid: dict,
        scorer: Callable,
        is_scorer_loss: bool,
        selected_params: Optional[dict],
    ) -> OptimizeGridSearch:
        instance = cls(model, param_grid, scorer, is_scorer_loss)
        instance.selected_params = selected_params
        return instance

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        if self.selected_params is None:
            scores = [self.scorer(y, result) for result in results]
            selected_index = (
                scores.index(min(scores))
                if self.is_scorer_loss
                else scores.index(max(scores))
            )
            self.selected_pipeline = [self.pipelines[selected_index]]
            return results[selected_index]
        else:
            return results[0]

    def get_child_transformations_primary(self) -> Pipelines:
        if self.selected_params is not None:
            self.model[0].set_params(**self.selected_params)
            return self.model
        else:
            grid = ParameterGrid(self.param_grid)
            for params in grid:
                self.model[0].set_params(**params)
                yield self.model

    def clone(self, clone_child_transformations: Callable) -> OptimizeGridSearch:
        return OptimizeGridSearch.from_cloned_instance(
            model=clone_child_transformations(self.model),
            param_grid=self.param_grid,
            scorer=self.scorer,
            is_scorer_loss=self.is_scorer_loss,
            selected_params=self.selected_params,
        )
