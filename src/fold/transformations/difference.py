# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from ..base import Artifact, InvertibleTransformation, Transformation, Tunable
from ..utils.dataframe import fill_na_inf, take_log, to_series


class Difference(InvertibleTransformation, Tunable):
    """
    Performs differencing.
    Sesonal differencing can be achieved by setting `lag` to the seasonality of the data.
    To achieve second-order differencing, simply chain multiple `Difference` transformations.

    Parameters
    ----------
    lag : int, optional
        the seasonality of the data, by default 1

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import Difference
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data(freq="min")
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = Difference()
    >>> X["sine"].head()
    2021-12-31 07:20:00    0.0000
    2021-12-31 07:21:00    0.0126
    2021-12-31 07:22:00    0.0251
    2021-12-31 07:23:00    0.0377
    2021-12-31 07:24:00    0.0502
    Freq: T, Name: sine, dtype: float64
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds["sine"].head()
    2021-12-31 15:40:00    0.0126
    2021-12-31 15:41:00    0.0126
    2021-12-31 15:42:00    0.0125
    2021-12-31 15:43:00    0.0126
    2021-12-31 15:44:00    0.0125
    Freq: T, Name: sine, dtype: float64

    ```

    References
    ----------

    [Stationarity and differencing](https://otexts.com/fpp2/stationarity.html)
    """

    first_values_X: Optional[Union[pd.DataFrame, pd.Series]] = None
    last_values_X: Optional[Union[pd.DataFrame, pd.Series]] = None

    def __init__(
        self,
        lag: int = 1,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.lag = lag
        self.params_to_try = params_to_try
        self.name = name or "Difference"
        self.properties = InvertibleTransformation.Properties(requires_X=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        self.last_values_X = X.iloc[-self.lag : None]
        self.first_values_X = X.iloc[: self.lag]

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        self.last_values_X = X.iloc[-self.lag : None]

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if in_sample:
            return X.diff(self.lag).fillna(0.0)
        else:
            return (
                pd.concat([self.last_values_X.copy(), X], copy=False, axis="index")
                .diff(self.lag)
                .iloc[self.lag :]
                .fillna(0.0)
            )

    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        if in_sample:
            X = X.copy()
            X.iloc[: self.lag] = to_series(self.first_values_X)
            for i in range(self.lag):
                X.iloc[i :: self.lag] = X.iloc[i :: self.lag].cumsum()
            return X
        else:
            X = pd.concat(
                [to_series(self.last_values_X).copy(), X], copy=False, axis="index"
            )
            for i in range(self.lag):
                X.iloc[i :: self.lag] = X.iloc[i :: self.lag].cumsum()
            return X.iloc[self.lag :]


class TakeReturns(Transformation, Tunable):
    """
    Takes the returns (percentage change between the current and a prior element).

    Parameters
    ----------
    log_returns : bool, optional, default False.
        If True, computes the log returns instead of the simple returns, default False.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import TakeReturns
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data(freq="min")
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = TakeReturns()
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> X["sine"].loc[preds.index].head()
    2021-12-31 15:40:00   -0.0000
    2021-12-31 15:41:00    0.0126
    2021-12-31 15:42:00    0.0251
    2021-12-31 15:43:00    0.0377
    2021-12-31 15:44:00    0.0502
    Freq: T, Name: sine, dtype: float64
    >>> preds["sine"].head()
    2021-12-31 15:40:00   -1.000000
    2021-12-31 15:41:00        -inf
    2021-12-31 15:42:00    0.992063
    2021-12-31 15:43:00    0.501992
    2021-12-31 15:44:00    0.331565
    Freq: T, Name: sine, dtype: float64

    ```

    """

    last_values_X: Optional[Union[pd.DataFrame, pd.Series]] = None

    def __init__(
        self,
        log_returns: bool = False,
        fillna: bool = False,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.log_returns = log_returns
        self.fillna = fillna
        self.params_to_try = params_to_try
        description = "log" if log_returns else "arithmetic"
        self.name = name or f"TakeReturns-{description}"
        self.properties = InvertibleTransformation.Properties(requires_X=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        self.last_values_X = X.iloc[-1:None]

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        self.last_values_X = X.iloc[-1:None]

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        def operation(df):
            if self.log_returns:
                return take_log(df / df.shift(1))
            else:
                return df.pct_change()

        fill_na = fill_na_inf if self.fillna else lambda x: x

        if in_sample:
            return fill_na(operation(X))
        else:
            return fill_na(
                operation(
                    pd.concat([self.last_values_X.copy(), X], copy=False, axis="index")
                )
            ).iloc[1:]
