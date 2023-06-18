# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from fold.utils.list import wrap_in_list

from ..base import Artifact, Transformation, Tunable, fit_noop

ColumnOrColumns = Union[str, List[str]]
ColumnFunction = Tuple[ColumnOrColumns, Callable]


class ApplyFunction(Transformation, Tunable):
    """
    Applies a function to one or more columns.

    Parameters
    ----------
    column_func
        A tuple of a column or list of columns and a function to apply to them.
        Can be a Callable in which case the function will be applied to all columns.
    name
        Name of the transformation.
    params_to_try
        Dictionary of parameters to try when tuning.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import ApplyFunction
    >>> from fold.models.dummy import DummyClassifier
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = ApplyFunction([("sine", np.square)])
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                               sine     sine_square
    2021-12-31 15:40:00       -0.0000   0.0000
    2021-12-31 15:41:00        0.0126   0.0159
    2021-12-31 15:42:00        0.0251   0.0006
    2021-12-31 15:43:00        0.0377   0.0014
    2021-12-31 15:44:00        0.0502   0.0025

    ```


    """

    def __init__(
        self,
        column_func: Union[Callable, ColumnFunction, List[ColumnFunction]],
        past_window_size: Optional[int] = None,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        if isinstance(column_func, Callable):
            self.column_func = [(["all"], column_func)]
        else:
            self.column_func = [
                (wrap_in_list(column), function)
                for column, function in wrap_in_list(column_func)
            ]
        self.properties = Transformation.Properties(
            requires_X=True, memory_size=past_window_size
        )
        self.params_to_try = params_to_try
        self.name = name or f"FunctionOnColumns_{self.column_func}"

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        X_function_applied = pd.DataFrame([], index=X.index)
        for columns, function in self.column_func:
            function_name = (
                function.__name__ if function.__name__ != "<lambda>" else "transformed"
            )

            if columns[0] == "all":
                columns = X.columns

            X_function_applied = pd.concat(
                [
                    X_function_applied,
                    function(X[columns].add_suffix(f"_{function_name}")),
                ],
                axis="columns",
            )
        return pd.concat([X, X_function_applied], axis="columns"), None

    fit = fit_noop
    update = fit
