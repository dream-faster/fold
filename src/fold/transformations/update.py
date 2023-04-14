from typing import Optional

import pandas as pd

from ..base import Transformation, fit_noop


class DontUpdate(Transformation):
    properties = Transformation.Properties(requires_X=False)

    def __init__(self, transformation: Transformation) -> None:
        self.transformation = transformation
        self.name = f"DontUpdate-{transformation.name}"

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.transformation.fit(X, y, sample_weights)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return self.transformation.transform(X, in_sample)

    update = fit_noop
