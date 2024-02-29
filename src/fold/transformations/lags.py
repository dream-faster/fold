# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


import pandas as pd
from finml_utils.dataframes import concat_on_columns

from ..base import Transformation, Tunable, feature_name_separator, fit_noop
from ..utils.checks import get_column_names
from ..utils.dataframe import to_dataframe
from ..utils.list import flatten, transform_range_to_list, wrap_in_list


class AddLagsX(Transformation, Tunable):
    """
    Adds past values of `X` for the desired column(s).

    Parameters
    ----------
    columns_and_lags : list[ColumnAndLag], ColumnAndLag
        A tuple (or a list of tuples) of the column name and a single or a list of lags to add as features.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import AddLagsX
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = AddLagsX([("sine", 1), ("sine", [2,3])])
    >>> preds, trained_pipeline, _, _ = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                           sine  sine~lag_1  sine~lag_2  sine~lag_3
    2021-12-31 15:40:00 -0.0000     -0.0126     -0.0251     -0.0377
    2021-12-31 15:41:00  0.0126     -0.0000     -0.0126     -0.0251
    2021-12-31 15:42:00  0.0251      0.0126     -0.0000     -0.0126
    2021-12-31 15:43:00  0.0377      0.0251      0.0126     -0.0000
    2021-12-31 15:44:00  0.0502      0.0377      0.0251      0.0126

    ```
    """

    ColumnAndLag = tuple[str, int | list[int]]

    def __init__(
        self,
        columns_and_lags: list[ColumnAndLag] | ColumnAndLag,
        keep_original: bool = True,
        output_dtype: type | None = None,
        name: str | None = None,
        params_to_try: dict | None = None,
    ) -> None:
        self.columns_and_lags = wrap_in_list(columns_and_lags)
        self.params_to_try = params_to_try
        self.output_dtype = output_dtype

        def check_and_transform_if_needed(
            column_and_lag: AddLagsX.ColumnAndLag,
        ) -> AddLagsX.ColumnAndLag:
            column, lags = column_and_lag
            if (
                not isinstance(lags, int)
                and not isinstance(lags, list)
                and not isinstance(lags, range)
            ):
                raise ValueError("lags must be an int or a List or a range")
            lags = sorted(
                transform_range_to_list([lags] if isinstance(lags, int) else lags)
            )
            return column, lags

        self.columns_and_lags = list(
            map(check_and_transform_if_needed, self.columns_and_lags)
        )
        self.properties = Transformation.Properties(
            requires_X=True,
            memory_size=max(flatten([l for _, l in self.columns_and_lags])),  # noqa
            disable_memory=True,
        )
        self.keep_original = keep_original
        self.name = name or f"AddLagsX-{self.columns_and_lags}"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        def convert_dtype_if_needed(df: pd.DataFrame) -> pd.DataFrame:
            return df.astype(self.output_dtype) if self.output_dtype is not None else df

        lagged_columns = []
        for column, lags in self.columns_and_lags:
            for lag in lags:
                selected = to_dataframe(X[get_column_names(column, X)])
                lagged_columns.append(
                    selected.shift(lag)[-len(X) :].add_suffix(
                        f"{feature_name_separator}lag_{lag}"
                    )
                )
        to_concat = [X, *lagged_columns] if self.keep_original else lagged_columns
        return convert_dtype_if_needed(concat_on_columns(to_concat))

    fit = fit_noop
    update = fit_noop
