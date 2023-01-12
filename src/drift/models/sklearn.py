from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from drift.models.base import Model, ModelType
from drift.transformations.base import Transformation


class SKLearnModel(Model):

    name = "SKLearnModel"
    type = ModelType.Multivariate

    def __init__(self, model: BaseEstimator) -> None:
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
