from typing import Callable

import pandas as pd

from .base import Transformation, fit_noop


class FunctionTransformation(Transformation):
    properties = Transformation.Properties()

    def __init__(self, func: Callable) -> None:
        self.func = func
        self.name = func.__name__

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return self.func(X)

    fit = fit_noop
    update = fit
