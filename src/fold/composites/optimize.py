# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Union

import pandas as pd
from sklearn.model_selection import ParameterGrid

from ..base import (
    Artifact,
    Composite,
    Optimizer,
    Pipeline,
    Transformation,
    Tunable,
    get_concatenated_names,
    traverse,
    traverse_apply,
)
from ..splitters import SingleWindowSplitter
from ..transformations.dev import Identity
from ..utils.list import to_hierachical_dict, wrap_in_list


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
        splitter: SingleWindowSplitter = SingleWindowSplitter(0.7),
        name: Optional[str] = None,
    ) -> None:
        self.pipeline = wrap_in_list(pipeline)
        self.name = name or "OptimizeGridSearch-" + get_concatenated_names(
            self.pipeline
        )
        self.scorer = scorer
        self.is_scorer_loss = is_scorer_loss
        self.splitter = splitter

    @classmethod
    def from_cloned_instance(
        cls,
        pipeline: Pipeline,
        scorer: Callable,
        is_scorer_loss: bool,
        splitter: SingleWindowSplitter,
        candidates: Optional[List[Tunable]],
        selected_params_: Optional[dict],
        selected_pipeline_: Optional[Tunable],
        scores_: Optional[List[float]],
        name: Optional[str],
    ) -> OptimizeGridSearch:
        instance = cls(pipeline, scorer, is_scorer_loss, splitter)
        instance.candidates = candidates
        instance.selected_params_ = selected_params_
        instance.selected_pipeline_ = selected_pipeline_
        instance.scores_ = scores_
        instance.name = name
        return instance

    def get_candidates(self) -> Iterable[Pipeline]:
        if self.candidates is None:
            tunables = [
                i
                for i in traverse(self.pipeline)
                if isinstance(i, Tunable) and i.get_params_to_try() is not None
            ]

            param_grid = {
                f"{transformation.id}.{key}": value
                for transformation in tunables
                for key, value in _process_params(
                    transformation.get_params_to_try()
                ).items()
            }
            self.param_permutations = [
                to_hierachical_dict(params) for params in ParameterGrid(param_grid)
            ]

            def __apply_params(params: dict) -> Callable:
                def __apply_params_to_transformation(
                    item: Union[Composite, Transformation], clone_children: Callable
                ) -> Union[Composite, Transformation]:
                    if not isinstance(item, Tunable):
                        return item
                    selected_params = params.get(item.id, {})
                    if "passthrough" in selected_params:
                        return Identity()  # type: ignore
                    return item.clone_with_params(
                        parameters={**item.get_params(), **selected_params},
                        clone_children=clone_children,
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

    def clone(self, clone_children: Callable) -> OptimizeGridSearch:
        return OptimizeGridSearch.from_cloned_instance(
            pipeline=clone_children(self.pipeline),
            scorer=self.scorer,
            is_scorer_loss=self.is_scorer_loss,
            splitter=self.splitter,
            candidates=self.candidates,
            selected_params_=self.selected_params_,
            selected_pipeline_=self.selected_pipeline_,
            scores_=self.scores_,
            name=self.name,
        )


def _process_params(params_to_try: dict) -> dict:
    if "passthrough" in params_to_try:
        params_to_try["passthrough"] = [True, False]
    return params_to_try
