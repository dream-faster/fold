from typing import List, Optional

import numpy as np

from ..transformations.base import Transformation
from .base import Model, ModelType


class Ensemble(Model):
    def __init__(self, models: List[Model], type: ModelType) -> None:
        self.models = models
        self.type = type
        self.name = "Ensemble-" + "-".join([model.name for model in models])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # to be done in the training loop with get_child_transformations()
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean(np.vstack([model.predict(X) for model in self.models]), axis=0)

    def get_child_transformations(self) -> Optional[List[Transformation]]:
        return self.models