from __future__ import annotations

from abc import abstractmethod
from enum import Enum

import pandas as pd

from ..transformations.base import Transformation


class ModelType(Enum):
    Univariate = 1
    Multivariate = 2


class Model(Transformation):

    name: str
    type: ModelType

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.predict(X).to_frame()
