# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import pandas as pd

from fold.splitters import SingleWindowSplitter

from ..base import Composite, Optimizer, Pipeline, Pipelines
from .common import get_concatenated_names


class SelectBestComposite(Composite):
    """
    Don't use this just yet. Coming later.
    """

    properties = Composite.Properties(primary_requires_predictions=True)
    selected_pipeline: Optional[Pipeline] = None

    def __init__(
        self,
        pipelines: Pipelines,
        scorer: Callable,
        is_scorer_loss: bool = True,
    ) -> None:
        self.pipelines = pipelines
        self.name = "SelectBest-" + get_concatenated_names(pipelines)
        self.scorer = scorer
        self.is_scorer_loss = is_scorer_loss

    @classmethod
    def from_cloned_instance(
        cls,
        pipelines: Pipelines,
        scorer: Callable,
        is_scorer_loss: bool,
        selected_pipeline: Optional[Pipeline],
    ) -> SelectBest:
        instance = cls(pipelines, scorer, is_scorer_loss)
        instance.selected_pipeline = selected_pipeline
        return instance

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        if self.selected_pipeline is None:
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
        if self.selected_pipeline is None:
            return self.pipelines
        else:
            return self.selected_pipeline

    def clone(self, clone_child_transformations: Callable) -> SelectBest:
        return SelectBestComposite.from_cloned_instance(
            pipelines=clone_child_transformations(self.pipelines),
            scorer=self.scorer,
            is_scorer_loss=self.is_scorer_loss,
            selected_pipeline=clone_child_transformations(self.selected_pipeline)
            if self.selected_pipeline is not None
            else None,
        )


class SelectBest(Optimizer):
    splitter: SingleWindowSplitter(train_window=0.5)  # type: ignore
    selected_pipeline: Optional[Pipeline] = None

    def __init__(
        self,
        pipelines: Pipelines,
        scorer: Callable,
        is_scorer_loss: bool = True,
    ) -> None:
        self.pipelines = pipelines
        self.name = "SelectBest-" + get_concatenated_names(pipelines)
        self.scorer = scorer
        self.is_scorer_loss = is_scorer_loss

    @classmethod
    def from_cloned_instance(
        cls,
        pipelines: Pipelines,
        scorer: Callable,
        is_scorer_loss: bool,
        selected_pipeline: Optional[Pipeline],
    ) -> SelectBest:
        instance = cls(pipelines, scorer, is_scorer_loss)
        instance.selected_pipeline = selected_pipeline
        return instance

    def get_candidates(self) -> Iterable["Pipeline"]:
        return self.pipelines

    def get_optimized_pipeline(self) -> Optional["Pipeline"]:
        return self.selected_pipeline

    def process_candidate_results(self, results: List[pd.DataFrame], y: pd.Series):
        if self.selected_pipeline is not None:
            raise ValueError("Optimizer is already fitted.")

        scores = [self.scorer(y, result) for result in results]
        selected_index = (
            scores.index(min(scores))
            if self.is_scorer_loss
            else scores.index(max(scores))
        )
        self.selected_pipeline = [self.pipelines[selected_index]]

    def clone(self, clone_child_transformations: Callable) -> SelectBest:
        return SelectBestComposite.from_cloned_instance(
            pipelines=clone_child_transformations(self.pipelines),
            scorer=self.scorer,
            is_scorer_loss=self.is_scorer_loss,
            selected_pipeline=clone_child_transformations(self.selected_pipeline)
            if self.selected_pipeline is not None
            else None,
        )
