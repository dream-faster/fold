from typing import Optional

import pandas as pd

from .base import Transformation


class Identity(Transformation):
    properties = Transformation.Properties()

    name = "Identity"

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        pass

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X
