# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


import numpy as np
import pandas as pd
from finml_utils.dataframes import concat_on_columns

from ..base import Artifact, InvertibleTransformation, Tunable, fit_noop


class TakeLog(InvertibleTransformation, Tunable):
    """
    Takes the logarithm of the data.

    Parameters
    ----------
    base : int, str, optional
        The base of the logarithm, by default "e".
        Valid values are "e", np.e, "10", 10, "2", 2.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import TakeLog
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data(freq="min")
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = TakeLog()
    >>> X["sine"].head()
    2021-12-31 07:20:00    0.0000
    2021-12-31 07:21:00    0.0126
    2021-12-31 07:22:00    0.0251
    2021-12-31 07:23:00    0.0377
    2021-12-31 07:24:00    0.0502
    Freq: T, Name: sine, dtype: float64
    >>> preds, trained_pipeline, _, _ = train_backtest(pipeline, X, y, splitter)
    >>> preds["sine"].head()
    2021-12-31 15:40:00        -inf
    2021-12-31 15:41:00   -4.374058
    2021-12-31 15:42:00   -3.684887
    2021-12-31 15:43:00   -3.278095
    2021-12-31 15:44:00   -2.991740
    Freq: T, Name: sine, dtype: float64

    ```
    """

    def __init__(
        self,
        base: int | str = "e",
        name: str | None = None,
        params_to_try: dict | None = None,
    ) -> None:
        if base not in ["e", np.e, "10", 10, "2", 2]:
            raise ValueError("base should be either 'e', np.e, '10', 10, '2', 2.")
        self.base = base
        self.params_to_try = params_to_try
        self.name = name or f"TakeLog-{self.base}"
        self.properties = InvertibleTransformation.Properties(requires_X=True)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.base in ("e", np.e):
            return pd.DataFrame(np.log(X.values), columns=X.columns, index=X.index)
        if self.base in ("10", 10):
            return pd.DataFrame(np.log10(X.values), columns=X.columns, index=X.index)
        if self.base in ("2", 2):
            return pd.DataFrame(np.log2(X.values), columns=X.columns, index=X.index)
        raise ValueError(f"Invalid base: {self.base}")

    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        if self.base in ("e", np.e):
            return pd.Series(np.exp(X.values), index=X.index)
        if self.base in ("10", 10):
            return 10**X
        if self.base in ("2", 2):
            return 2**X
        raise ValueError(f"Invalid base: {self.base}")

    fit = fit_noop
    update = fit_noop


class TurnPositive(InvertibleTransformation):
    """
    Adds a constant to the data, varying by column, so that all values are positive.
    It identifies the constant during training, and applies it during inference (and backtesting).
    Therefore there's no guarantee that the data will be positive during inference (and backtesting).

    It can not be updated after the initial training, as that'd change the underlying distribution of the data.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import TurnPositive
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data(freq="min")
    >>> X, y  = X - 1, y - 1
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = TurnPositive()
    >>> X["sine"].head()
    2021-12-31 07:20:00   -1.0000
    2021-12-31 07:21:00   -0.9874
    2021-12-31 07:22:00   -0.9749
    2021-12-31 07:23:00   -0.9623
    2021-12-31 07:24:00   -0.9498
    Freq: T, Name: sine, dtype: float64
    >>> preds, trained_pipeline, _, _ = train_backtest(pipeline, X, y, splitter)
    >>> preds["sine"].head()
    2021-12-31 15:40:00    2.0000
    2021-12-31 15:41:00    2.0126
    2021-12-31 15:42:00    2.0251
    2021-12-31 15:43:00    2.0377
    2021-12-31 15:44:00    2.0502
    Freq: T, Name: sine, dtype: float64

    ```
    """

    name = "TurnPositive"
    properties = InvertibleTransformation.Properties(requires_X=True)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        raw_y: pd.Series | None = None,
    ) -> Artifact | None:
        min_values = X.min(axis=0)
        self.constant = dict(min_values[min_values <= 0].abs() + 1)
        self.properties = InvertibleTransformation.Properties(requires_X=True)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        transformed_columns = X[list(self.constant.keys())] + pd.Series(self.constant)
        return concat_on_columns(
            [X.drop(columns=self.constant.keys()), transformed_columns],
        )

    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        return X - next(iter(self.constant.values()))

    update = fit_noop


class MultiplyBy(InvertibleTransformation, Tunable):
    """
    Multiplies the data by a constant.
    """

    constant: float

    def __init__(
        self,
        constant: float,
        name: str | None = None,
        params_to_try: dict | None = None,
    ) -> None:
        self.constant = constant
        self.params_to_try = params_to_try
        self.name = name or f"MultiplyBy-{constant}"
        self.properties = InvertibleTransformation.Properties(requires_X=True)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X * self.constant

    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        return X / self.constant

    fit = fit_noop
    update = fit_noop
