# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import pandas as pd
from sklearn.model_selection import ParameterGrid

from fold.utils.list import wrap_in_list

from ..base import Artifact, Optimizer, Tunable
from .common import get_concatenated_names


class OptimizeGridSearch(Optimizer):
    candidates: Optional[List[Tunable]] = None
    selected_params_: Optional[dict] = None
    selected_model_: Optional[Tunable] = None
    scores_: Optional[List[float]] = None
    param_permutations: List[dict]

    def __init__(
        self,
        model: Tunable,
        param_grid: dict,
        scorer: Callable,
        is_scorer_loss: bool = True,
    ) -> None:
        self.model = wrap_in_list(model)
        self.param_grid = param_grid
        self.name = "GridSearchOptimizer-" + get_concatenated_names(self.model)
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
        candidates: Optional[List[Tunable]],
        selected_params_: Optional[dict],
        selected_model_: Optional[Tunable],
        scores_: Optional[List[float]],
    ) -> OptimizeGridSearch:
        instance = cls(model, param_grid, scorer, is_scorer_loss)
        instance.candidates = candidates
        instance.selected_params_ = selected_params_
        instance.selected_model_ = selected_model_
        instance.scores_ = scores_
        return instance

    def get_candidates(self) -> Iterable[Tunable]:
        if self.candidates is None:
            self.candidates = [
                self.model[0].clone_with_params(
                    **{**self.model[0].get_params(), **params}
                )
                for params in self.param_permutations
            ]
        return self.candidates

    def get_optimized_pipeline(self) -> Optional[Tunable]:
        return self.selected_model_

    def process_candidate_results(
        self, results: List[pd.DataFrame], y: pd.Series
    ) -> Optional[Artifact]:
        if self.selected_params_ is not None:
            raise ValueError("Optimizer is already fitted.")

        scores = [self.scorer(y[-len(result) :], result) for result in results]
        selected_index = (
            scores.index(min(scores))
            if self.is_scorer_loss
            else scores.index(max(scores))
        )
        self.selected_params_ = self.param_permutations[selected_index]
        self.selected_model_ = self.candidates[selected_index]
        # TODO: need to store the params for each score as well
        self.scores_ = scores
        return pd.DataFrame(
            {"selected_params": [self.selected_params_]}, index=y.index[-1:]
        )

    def clone(self, clone_child_transformations: Callable) -> OptimizeGridSearch:
        return OptimizeGridSearch.from_cloned_instance(
            model=clone_child_transformations(self.model),
            param_grid=self.param_grid,
            scorer=self.scorer,
            is_scorer_loss=self.is_scorer_loss,
            candidates=self.candidates,
            selected_params_=self.selected_params_,
            selected_model_=self.selected_model_,
            scores_=self.scores_,
        )
