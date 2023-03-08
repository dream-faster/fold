from inspect import getfullargspec
from typing import Callable, Optional

import pandas as pd

from .base import Transformation


class Test(Transformation):
    properties = Transformation.Properties()
    __test__ = False

    def __init__(
        self,
        fit_func: Callable,
        transform_func: Callable,
        update_func: Optional[Callable] = None,
        inverse_transform_func: Optional[Callable] = None,
    ) -> None:
        self.name = "Test"
        self.fit_func = fit_func
        self.transform_func = transform_func
        self.update_func = update_func
        self.inverse_transform_func = inverse_transform_func

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        argspec = getfullargspec(self.fit_func)
        if len(argspec.args) == 1:
            self.fit_func(X)
        elif len(argspec.args) == 2:
            self.fit_func(X, y)
        elif len(argspec.args) == 3:
            self.fit_func(X, y, sample_weights)
        else:
            raise ValueError(
                "fit_func must accept between 1 and 3 arguments, "
                f"but {len(argspec.args)} were given."
            )

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if self.update_func is None:
            return
        argspec = getfullargspec(self.update_func)
        if len(argspec.args) == 1:
            self.update_func(X)
        elif len(argspec.args) == 2:
            self.update_func(X, y)
        elif len(argspec.args) == 3:
            self.update_func(X, y, sample_weights)
        else:
            raise ValueError(
                "update_func must accept between 1 and 3 arguments, "
                f"but {len(argspec.args)} were given."
            )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return_value = self.transform_func(X)
        if return_value is None:
            return X
        return return_value

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.inverse_transform_func is not None:
            self.inverse_transform_func(X)
        return X
