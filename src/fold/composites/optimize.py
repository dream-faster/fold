# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional

import pandas as pd
from sklearn.model_selection import ParameterGrid

from fold.utils.list import wrap_in_list

from ..base import Composite, Optimizer, Pipelines, Tunable
from .common import get_concatenated_names


class OptimizeGridSearch(Composite):
    """
    Don't use this just yet. Coming later.
    """

    properties = Composite.Properties(primary_requires_predictions=True)
    selected_params: Optional[dict] = None

    def __init__(
        self,
        model: Tunable,
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
        model: Tunable,
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


class SelectGridSearch(Optimizer):
    properties = Composite.Properties(primary_requires_predictions=True)
    selected_params: Optional[dict] = None
    param_permutations: List[dict(str, Any)]

    def __init__(
        self,
        model: Tunable,
        param_grid: dict,
        scorer: Callable,
        is_scorer_loss: bool = True,
    ) -> None:
        self.model = wrap_in_list(model)
        self.param_grid = param_grid
        self.name = "SelectGridSearch-" + get_concatenated_names(self.model)
        self.scorer = scorer
        self.is_scorer_loss = is_scorer_loss
        self.param_permutations = list(ParameterGrid(self.param_grid))

    @classmethod
    def from_cloned_instance(
        cls,
        model: Tunable,
        param_grid: dict,
        scorer: Callable,
        is_scorer_loss: bool,
        selected_params: Optional[dict],
    ) -> SelectGridSearch:
        instance = cls(model, param_grid, scorer, is_scorer_loss)
        instance.selected_params = selected_params
        return instance

    def get_candidates(self) -> Iterable["Tunable"]:
        models = [deepcopy(self.model[0]) for _ in self.param_permutations]

        for model, params in zip(models, self.param_permutations):
            model.set_params(**params)

        return models

    def get_optimized_pipeline(self) -> Optional["Tunable"]:
        if self.selected_params is None:
            return None
        self.model[0].set_params(**self.selected_params)
        return self.model[0]

    def process_candidate_results(self, results: List[pd.DataFrame], y: pd.Series):
        if self.selected_params is not None:
            raise ValueError("Optimizer is already fitted.")

        scores = [self.scorer(y, result) for result in results]
        selected_index = (
            scores.index(min(scores))
            if self.is_scorer_loss
            else scores.index(max(scores))
        )
        self.selected_params = self.param_permutations[selected_index]

    def clone(self, clone_child_transformations: Callable) -> SelectGridSearch:
        return SelectGridSearch.from_cloned_instance(
            model=clone_child_transformations(self.model),
            param_grid=self.param_grid,
            scorer=self.scorer,
            is_scorer_loss=self.is_scorer_loss,
            selected_params=self.selected_params,
        )
