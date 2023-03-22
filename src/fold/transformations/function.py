from typing import Callable, Optional

import pandas as pd

from .base import Transformation, fit_noop


class WrapFunction(Transformation):
    """
    Wraps and arbitrary function that will run at inference.
    """

    def __init__(self, func: Callable, past_window_size: Optional[int]) -> None:
        self.func = func
        self.name = func.__name__
        self.properties = Transformation.Properties(memory_size=past_window_size)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return self.func(X)

    fit = fit_noop
    update = fit
