# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from fold.utils.list import wrap_in_list

from ..base import Artifact, Transformation, Tunable, fit_noop

ColumnOrColumns = Union[str, List[str]]
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


class ApplyFunction(Transformation):
    """
    Wraps and arbitrary function that will run at inference.
    """

    def __init__(self, func: Callable, past_window_size: Optional[int]) -> None:
        self.func = func
        self.name = func.__name__
        self.properties = Transformation.Properties(
            requires_X=True, memory_size=past_window_size
        )

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        return self.func(X), None

    fit = fit_noop
    update = fit
