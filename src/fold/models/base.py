from abc import abstractmethod
from typing import Union

import pandas as pd

from ..transformations.base import Transformation


class Model(Transformation):
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pred = self.predict(X)
        if isinstance(pred, pd.Series):
            return pred.to_frame()
        else:
            return pred
