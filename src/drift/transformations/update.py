from typing import Optional

import pandas as pd

from .base import Transformation


class DontUpdate(Transformation):

    properties = Transformation.Properties(requires_past_X=True)
    number_of_fits = 0

    def __init__(self, transformation: Transformation) -> None:
        self.transformation = transformation
        self.name = f"DontUpdate-{transformation.name}"

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if self.number_of_fits == 0:
            self.transformation.fit(X, y, sample_weights)
            self.number_of_fits += 1
        else:
            pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.transformation.transform(X)
