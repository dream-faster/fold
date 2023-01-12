from typing import List, Optional

import numpy as np

from .base import Transformation


class Ensemble(Transformation):
    def __init__(self, models: List[Transformation]) -> None:
        self.models = models
        self.name = "Ensemble-" + "-".join([model.name for model in models])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # to be done in the training loop with get_child_transformations()
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean(np.vstack([model.transform(X) for model in self.models]), axis=0)

    def get_child_transformations(self) -> Optional[List[Transformation]]:
        return self.models
