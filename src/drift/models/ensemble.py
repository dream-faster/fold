from typing import List

import numpy as np

from .base import Model, ModelType


class Ensemble(Model):
    def __init__(self, models: List[Model], type: ModelType) -> None:
        self.models = models
        self.type = type
        self.name = "Ensemble-" + "-".join([model.name for model in models])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean(np.vstack([model.predict(X) for model in self.models]), axis=0)
