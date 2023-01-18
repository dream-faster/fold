from typing import List

import pandas as pd

from ..transformations.base import Composite, Transformations


class Ensemble(Composite):
    def __init__(self, models: Transformations) -> None:
        self.models = models
        self.name = "Ensemble-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in models
            ]
        )

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(results, axis=1).mean(axis=0).to_frame()

    def get_child_transformations(self) -> List[Transformations]:
        return self.models

    def set_child_transformations(self, transformations: Transformations) -> None:
        self.models = transformations
