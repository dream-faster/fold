from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Transformation


class Sampling(Transformation):
    properties = Transformation.Properties()

    def __init__(
        self,
        sampler,
        model: Transformation,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.name = f"Sampling-{sampler.__class__.__name__}-{model.name}"

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        X, y = self.sampler.fit_resample(X, y)
        self.model.fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)
