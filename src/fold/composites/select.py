from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd

from ..transformations.base import Transformations, TransformationsAlwaysList
from ..transformations.common import get_concatenated_names
from .base import Composite


class SelectBest(Composite):
    """
    Don't use this just yet. Coming in v0.2.
    """

    properties = Composite.Properties()

    def __init__(
        self,
        models: TransformationsAlwaysList,
        scorer: Callable,
        is_scorer_loss: bool = True,
        selected_model: Optional[Transformations] = None,
    ) -> None:
        self.models = models
        self.name = "SelectBest-" + get_concatenated_names(models)
        self.scorer = scorer
        self.is_scorer_loss = is_scorer_loss
        self.selected_model = selected_model

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        if self.selected_model is None:
            scores = [self.scorer(y, result) for result in results]
            selected_index = (
                scores.index(min(scores))
                if self.is_scorer_loss
                else scores.index(max(scores))
            )
            self.selected_model = [self.models[selected_index]]
            return results[selected_index]
        else:
            return results[0]

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        if self.selected_model is None:
            return self.models
        else:
            return self.selected_model

    def clone(self, clone_child_transformations: Callable) -> SelectBest:
        return SelectBest(
            models=clone_child_transformations(self.models),
            scorer=self.scorer,
            is_scorer_loss=self.is_scorer_loss,
            selected_model=clone_child_transformations(self.selected_model)
            if self.selected_model is not None
            else None,
        )
