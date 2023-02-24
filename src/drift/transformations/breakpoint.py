from typing import Callable, Optional

import pandas as pd

from .base import Transformation


class Breakpoint(Transformation):

    properties = Transformation.Properties(requires_past_X=True)

    def __init__(self, stop_at_fit: bool, stop_at_transform: bool) -> None:
        self.name = "Debug"
        self.stop_at_fit = stop_at_fit
        self.stop_at_transform = stop_at_transform

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if self.stop_at_fit:
            breakpoint()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.stop_at_transform:
            breakpoint()
        return X
