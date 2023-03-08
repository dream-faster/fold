from typing import Optional

import pandas as pd

from .base import Transformation


class Breakpoint(Transformation):
    properties = Transformation.Properties()

    def __init__(
        self,
        stop_at_fit: bool = True,
        stop_at_update: bool = True,
        stop_at_transform: bool = True,
    ) -> None:
        self.name = "Debug"
        self.stop_at_fit = stop_at_fit
        self.stop_at_update = stop_at_update
        self.stop_at_transform = stop_at_transform

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if self.stop_at_fit:
            breakpoint()

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if self.stop_at_update:
            breakpoint()

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.stop_at_transform:
            breakpoint()
        return X
