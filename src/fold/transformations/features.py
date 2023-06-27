# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from ..base import Artifact, Transformation, Tunable, fit_noop
from ..utils.enums import ParsableEnum
from ..utils.list import wrap_in_list


class PredefinedFunction(ParsableEnum):
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


ColumnOrColumns = Union[str, List[str]]
FunctionOrPredefined = Union[Callable, PredefinedFunction, str]
ColumnWindowFunction = Tuple[ColumnOrColumns, Optional[int], FunctionOrPredefined]


class AddWindowFeatures(Transformation, Tunable):

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
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
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

    def __init__(
        self,
        column_window_func: Union[ColumnWindowFunction, List[ColumnWindowFunction]],
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        def replace_nan(value: Optional[int], replacement: int = 0) -> int:
            return value if value is not None else replacement

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
        max_memory = max(
            [replace_nan(window) for _, window, _ in self.column_window_func]
        )
        self.properties = Transformation.Properties(
            requires_X=True, memory_size=max_memory
        )
        self.params_to_try = params_to_try
        self.name = name or "AddWindowFeatures"

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        def apply_function(
            columns: List[str],
            window: int,
            function: Callable,
        ):
            if isinstance(function, PredefinedFunction):
                function = getattr(pd.core.window.rolling.Rolling, function.value)
            function_name = (
                function.__name__ if function.__name__ != "<lambda>" else "transformed"
            )

            if columns[0] == "all":
                columns = X.columns

            if window is None:
                return function(
                    X[columns].add_suffix(f"_{window}_{function_name}").expanding()
                )
            else:
                return function(
                    X[columns].add_suffix(f"_{window}_{function_name}").rolling(window)
                )

        X_function_applied = [
            apply_function(columns, window, function)
            for columns, window, function in self.column_window_func
        ]

        return pd.concat([X] + X_function_applied, axis="columns"), None

    fit = fit_noop
    update = fit


ColumnFunction = Tuple[ColumnOrColumns, Callable]


class AddFeatures(Transformation, Tunable):
    """
    Applies a function to one or more columns.

    Parameters
    ----------
    column_func: Union[Callable, Tuple[Union[str, List[str]], Callable]]
        A tuple of a column or list of columns and a function to apply to them.
    name: str
        Name of the transformation.
    params_to_try: dict
        Dictionary of parameters to try when tuning.

    Returns
    ----------
    Tuple[pd.DataFrame, Optional[Artifact]]: returns the transformed DataFrame with the original dataframe concatinated.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import AddFeatures
    >>> from fold.models.dummy import DummyClassifier
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> import numpy as np
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = AddFeatures([("sine", np.square)])
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                           sine  sine_square
    2021-12-31 15:40:00 -0.0000     0.000000
    2021-12-31 15:41:00  0.0126     0.000159
    2021-12-31 15:42:00  0.0251     0.000630
    2021-12-31 15:43:00  0.0377     0.001421
    2021-12-31 15:44:00  0.0502     0.002520

    ```


    """

    def __init__(
        self,
        column_func: Union[ColumnFunction, List[ColumnFunction]],
        past_window_size: Optional[int] = None,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.column_func = [
            (wrap_in_list(column), function)
            for column, function in wrap_in_list(column_func)
        ]

        self.properties = Transformation.Properties(
            requires_X=True, memory_size=past_window_size
        )
        self.params_to_try = params_to_try
        self.name = name or f"ApplyFunction_{self.column_func}"

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        X_function_applied = []

        def apply_function(columns: List[str], function: Callable) -> pd.DataFrame:
            function_name = (
                function.__name__ if function.__name__ != "<lambda>" else "transformed"
            )

            if columns[0] == "all":
                columns = X.columns
            return function(X[columns].add_suffix(f"_{function_name}"))

        X_function_applied = [
            apply_function(columns, function) for columns, function in self.column_func
        ]

        return pd.concat([X] + X_function_applied, axis="columns"), None

    fit = fit_noop
    update = fit


class AddRollingCorrelation(Transformation, Tunable):
    def __init__(
        self,
        column_pairs: Union[Tuple[str], List[Tuple[str]]],
        window: int,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.column_pairs = wrap_in_list(column_pairs)
        assert all(
            len(pair) == 2 for pair in self.column_pairs
        ), "All column pairs must be of length 2"
        self.window = window
        self.properties = Transformation.Properties(requires_X=True, memory_size=window)
        self.params_to_try = params_to_try
        self.name = name or "AddRollingCorrelation"

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        def apply_function(
            column_pair: Tuple[str],
            window: int,
        ):
            lhs = column_pair[0]
            rhs = column_pair[1]
            return (
                X[lhs]
                .rolling(window)
                .corr(X[rhs])
                .rename(f"{lhs}_{rhs}_rolling_corr_{window}")
            )

        X_function_applied = [
            apply_function(column_pair, self.window)
            for column_pair in self.column_pairs
        ]

        return pd.concat([X] + X_function_applied, axis="columns"), None

    fit = fit_noop
    update = fit
