from abc import abstractmethod
from typing import Optional

import pandas as pd

from ..transformations.base import Transformation


class Model(Transformation):

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pred = self.predict(X)
        if isinstance(pred, pd.Series):
            return pred.to_frame()
        else:
            return pred
