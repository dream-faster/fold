from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd

from ..transformations.base import TransformationsAlwaysList
from ..transformations.common import get_concatenated_names
from .base import Composite
from .columns import postprocess_results


class Ensemble(Composite):
    properties = Composite.Properties()

    def __init__(self, models: TransformationsAlwaysList) -> None:
        self.models = models
        self.name = "Ensemble-" + get_concatenated_names(models)

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return postprocess_results(results, self.name)

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.models

    def clone(self, clone_child_transformations: Callable) -> Ensemble:
        return Ensemble(
            models=clone_child_transformations(self.models),
        )
