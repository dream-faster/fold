# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import pandas as pd
from sklearn.model_selection import ParameterGrid

from fold.traverse import traverse, traverse_apply
from fold.utils.list import ensure_dict, to_hierachical_dict, wrap_in_list

from ..base import Artifact, Optimizer, Pipeline, Tunable
from .common import get_concatenated_names


class OptimizeGridSearch(Optimizer):
    candidates: Optional[List[Pipeline]] = None
    selected_params_: Optional[dict] = None
    selected_pipeline_: Optional[Tunable] = None
    scores_: Optional[List[float]] = None
    param_permutations: List[dict]

    def __init__(
        self,
        pipeline: Pipeline,
        scorer: Callable,
        is_scorer_loss: bool = True,
    ) -> None:
        self.pipeline = wrap_in_list(pipeline)
        self.name = "GridSearchOptimizer-" + get_concatenated_names(self.pipeline)
        self.scorer = scorer
        self.is_scorer_loss = is_scorer_loss

    @classmethod
    def from_cloned_instance(
        cls,
        pipeline: Pipeline,
        scorer: Callable,
        is_scorer_loss: bool,
        candidates: Optional[List[Tunable]],
        selected_params_: Optional[dict],
        selected_pipeline_: Optional[Tunable],
        scores_: Optional[List[float]],
    ) -> OptimizeGridSearch:
        instance = cls(pipeline, scorer, is_scorer_loss)
        instance.candidates = candidates
        instance.selected_params_ = selected_params_
        instance.selected_pipeline_ = selected_pipeline_
        instance.scores_ = scores_
        return instance

    def get_candidates(self) -> Iterable[Pipeline]:
        if self.candidates is None:
            all_transformation = list(traverse(self.pipeline))

            param_grid = {
                f"{transformation.id}.{key}": value
                for transformation in all_transformation
                for key, value in ensure_dict(transformation.params_to_try).items()
                if transformation.params_to_try is not None
            }
            self.param_permutations = [
                to_hierachical_dict(params)
                for params in list(ParameterGrid(param_grid))
            ]

            def __apply_params(params: dict) -> Callable:
                def __apply_params_to_transformation(
                    transformation: Tunable,
                ) -> Tunable:
                    selected_params = params[transformation.id]
                    return transformation.clone_with_params(
                        **{**transformation.get_params(), **selected_params}
                    )

                return __apply_params_to_transformation

            self.candidates = [
                traverse_apply(self.pipeline, __apply_params(params))
                for params in self.param_permutations
            ]
        return self.candidates

    def get_optimized_pipeline(self) -> Optional[Tunable]:
        return self.selected_pipeline_

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
        self.selected_pipeline_ = self.candidates[selected_index]
        # TODO: need to store the params for each score as well
        self.scores_ = scores
        return pd.DataFrame(
            {"selected_params": [self.selected_params_]}, index=y.index[-1:]
        )

    def clone(self, clone_child_transformations: Callable) -> OptimizeGridSearch:
        return OptimizeGridSearch.from_cloned_instance(
            pipeline=clone_child_transformations(self.pipeline),
            scorer=self.scorer,
            is_scorer_loss=self.is_scorer_loss,
            candidates=self.candidates,
            selected_params_=self.selected_params_,
            selected_pipeline_=self.selected_pipeline_,
            scores_=self.scores_,
        )
