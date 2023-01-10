from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd


class ModelType(Enum):
    Univariate = 1
    Multivariate = 2


class Model(ABC):

    name: str
    type: ModelType

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
