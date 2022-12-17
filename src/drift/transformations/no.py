from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Transformation


class NoTransformation(Transformation):

    name = "NoTransformation"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def clone(self) -> NoTransformation:
        return self
