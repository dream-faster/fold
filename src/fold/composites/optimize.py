# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd
from sklearn.model_selection import ParameterGrid

from fold.base.scoring import score_results

from ..base import (
    Artifact,
    Optimizer,
    Pipeline,
    Tunable,
    get_concatenated_names,
    traverse_apply,
)
from ..splitters import SingleWindowSplitter
from ..utils.list import to_hierachical_dict, wrap_in_list
from .utils import _apply_params, _check_for_duplicate_names, _extract_param_grid

_divider = "¦"


class OptimizeGridSearch(Optimizer):
    candidates: Optional[List[Pipeline]] = None
    selected_params_: Optional[dict] = None
    selected_pipeline_: Optional[Tunable] = None
    scores_: Optional[List[float]] = None
    param_permutations: List[dict]

    def __init__(
        self,
        pipeline: Pipeline,
        krisi_metric_key: str,
        krisi_args: Optional[dict] = None,
        is_scorer_loss: bool = True,
        splitter: SingleWindowSplitter = SingleWindowSplitter(0.7),
        name: Optional[str] = None,
    ) -> None:
        self.pipeline = wrap_in_list(pipeline)
        self.name = name or "OptimizeGridSearch-" + get_concatenated_names(
            self.pipeline
        )
        self.krisi_metric_key = krisi_metric_key
        self.krisi_args = krisi_args or {}
        self.is_scorer_loss = is_scorer_loss
        self.splitter = splitter
        _check_for_duplicate_names(self.pipeline)

    @classmethod
    def from_cloned_instance(
        cls,
        pipeline: Pipeline,
        krisi_metric_key: str,
        krisi_args: Optional[dict],
        is_scorer_loss: bool,
        splitter: SingleWindowSplitter,
        candidates: Optional[List[Tunable]],
        selected_params_: Optional[dict],
        selected_pipeline_: Optional[Tunable],
        scores_: Optional[List[float]],
        name: Optional[str],
        id: str,
    ) -> OptimizeGridSearch:
        instance = cls(
            pipeline=pipeline,
            krisi_metric_key=krisi_metric_key,
            krisi_args=krisi_args,
            is_scorer_loss=is_scorer_loss,
            splitter=splitter,
        )
        instance.candidates = candidates
        instance.selected_params_ = selected_params_
        instance.selected_pipeline_ = selected_pipeline_
        instance.scores_ = scores_
        instance.name = name
        instance.id = id
        return instance

    def get_candidates(self) -> List[Pipeline]:
        if self.candidates is None:
            param_grid = _extract_param_grid(self.pipeline, _divider)

            self.param_permutations = [
                to_hierachical_dict(params, _divider)
                for params in ParameterGrid(param_grid)
            ]

            self.candidates = [
                traverse_apply(self.pipeline, _apply_params(params))
                for params in self.param_permutations
            ]
            return self.candidates
        else:
            return []

    def get_optimized_pipeline(self) -> Optional[Tunable]:
        return self.selected_pipeline_

    def process_candidate_results(
        self,
        results: List[pd.DataFrame],
        y: pd.Series,
        artifacts: List[pd.DataFrame],
    ) -> Optional[Artifact]:
        scores = [
            score_results(
                result,
                y=y[-len(result) :],
                artifacts=artifact,
                krisi_args=self.krisi_args,
            )[0][self.krisi_metric_key].result
            for result, artifact in zip(results, artifacts)
        ]
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
            krisi_metric_key=self.krisi_metric_key,
            krisi_args=self.krisi_args,
            is_scorer_loss=self.is_scorer_loss,
            splitter=self.splitter,
            candidates=self.candidates,
            selected_params_=self.selected_params_,
            selected_pipeline_=self.selected_pipeline_,
            scores_=self.scores_,
            name=self.name,
            id=self.id,
        )
