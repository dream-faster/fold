from __future__ import annotations

from typing import Optional

import pandas as pd

from ..models.base import Model
from .base import Transformation


class Sampling(Transformation):

    properties = Transformation.Properties(requires_past_X=False)

    def __init__(
        self,
        sampler,
        model: Model,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.name = f"Sampling-{sampler.__class__.__name__}-{model.name}"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        X, y = self.sampler.fit_resample(X, y)
        self.model.fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)
