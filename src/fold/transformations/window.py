# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from enum import Enum
from typing import Callable, List, Tuple, Union

import pandas as pd

from ..base import Transformation, fit_noop
from ..utils.list import wrap_in_list


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

    """
    Creates rolling window features on the specified columns.
    Equivalent to adding a new column by running: `df[column].rolling(window).function()`.


    Parameters
    ----------
    column_window_func : ColumnWindowFunction, List[ColumnWindowFunction]
        A list of tuples, where each tuple contains the column name, the window size and the function to apply.
        The function can be a predefined function (see PredefinedFunction) or a Callable (with a single parameter).

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import AddWindowFeatures
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(initial_train_window=0.5, step=0.2)
    >>> pipeline = AddWindowFeatures(("sine", 10, "mean"))
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                           sine  sine_10_mean
    2021-12-31 15:40:00 -0.0000      -0.05649
    2021-12-31 15:41:00  0.0126      -0.04394
    2021-12-31 15:42:00  0.0251      -0.03139
    2021-12-31 15:43:00  0.0377      -0.01883
    2021-12-31 15:44:00  0.0502      -0.00628

    ```
    """

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
                if isinstance(function, Callable)
                else PredefinedFunction.from_str(function),
            )
            for column, window, function in wrap_in_list(column_window_func)
        ]
        max_memory = max([window for _, window, _ in self.column_window_func])
        self.properties = Transformation.Properties(
            requires_X=True, memory_size=max_memory
        )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        X_function_applied = pd.DataFrame([], index=X.index)
        for columns, window, function in self.column_window_func:
            if isinstance(function, PredefinedFunction):
                function = getattr(pd.core.window.rolling.Rolling, function.value)
            function_name = (
                function.__name__ if function.__name__ != "<lambda>" else "transformed"
            )
            if columns[0] == "all":
                X_function_applied = pd.concat(
                    [
                        X_function_applied,
                        function(X.rolling(window)).add_suffix(
                            f"_{window}_{function_name}"
                        ),
                    ],
                    axis="columns",
                )
            else:
                for col in columns:
                    X_function_applied[f"{col}_{window}_{function_name}"] = function(
                        X[col].rolling(window)
                    )
        return pd.concat([X, X_function_applied], axis="columns")

    fit = fit_noop
    update = fit
