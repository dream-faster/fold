from typing import List, Optional

import numpy as np
import pandas as pd

from ..transformations.base import Transformation
from .base import Model


class Ensemble(Model):
    def __init__(self, models: List[Transformation]) -> None:
        self.models = models
        self.name = "Ensemble-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in models
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # to be done in the training loop with get_child_transformations()
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.mean(np.vstack([model.transform(X) for model in self.models]), axis=0)

    def get_child_transformations(self) -> Optional[List[Transformation]]:
        return self.models

    def set_child_transformations(self, transformations: List[Transformation]) -> None:
        self.models = transformations
