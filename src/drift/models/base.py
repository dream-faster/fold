from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class ModelType(Enum):
    Univariate = 1
    Multivariate = 2


class Model(ABC):

    name: str
    type: ModelType

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
