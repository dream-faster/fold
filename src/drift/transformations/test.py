from typing import Callable, Optional

import pandas as pd

from .base import Transformation


class TestIdentity(Transformation):

    properties = Transformation.Properties(requires_past_X=True)

    def __init__(self, fit_func: Callable, transform_func: Callable) -> None:
        self.name = "TestTransform"
        self.fit_func = fit_func
        self.transform_func = transform_func

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.fit_func(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.transform_func(X)
        return X
