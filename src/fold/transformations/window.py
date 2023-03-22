from __future__ import annotations

from enum import Enum
from typing import Callable, List, Tuple, Union

import pandas as pd

from ..utils.list import wrap_in_list
from .base import Transformation, fit_noop


class PredefinedFunction(Enum):
    mean = "mean"
    sum = "sum"
    median = "median"
    std = "std"
    var = "var"
    kurt = "kurt"
    min = "min"
    max = "max"
    corr = "corr"
    cov = "cov"
    skew = "skew"
    sem = "sem"

    @staticmethod
    def from_str(value: Union[str, PredefinedFunction]) -> PredefinedFunction:
        if isinstance(value, PredefinedFunction):
            return value
        for strategy in PredefinedFunction:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown PredefinedFunction: {value}")


ColumnOrColumns = Union[str, List[str]]
FunctionOrPredefined = Union[Callable, PredefinedFunction, str]
ColumnWindowFunction = Tuple[ColumnOrColumns, int, FunctionOrPredefined]


class AddWindowFeatures(Transformation):
    name = "AddWindowFeatures"

    def __init__(
        self,
        column_window_func: Union[ColumnWindowFunction, List[ColumnWindowFunction]],
    ) -> None:
        self.column_window_func = [
            (
                wrap_in_list(column),
                window,
                function
                if function is Callable
                else PredefinedFunction.from_str(function),
            )
            for column, window, function in wrap_in_list(column_window_func)
        ]
        max_memory = max([window for _, window, _ in self.column_window_func])
        self.properties = Transformation.Properties(memory=max_memory)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        X = X.copy()
        for columns, window, function in self.column_window_func:
            for col in columns:
                if isinstance(function, PredefinedFunction):
                    function = getattr(pd.core.window.rolling.Rolling, function.value)
                X[f"{col}_{window}_{function.__name__}"] = function(
                    X[col].rolling(window)
                )
        return X

    fit = fit_noop
    update = fit
