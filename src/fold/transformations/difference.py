from typing import Optional

import pandas as pd

from ..base import InvertibleTransformation


class Difference(InvertibleTransformation):
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
    >>> splitter = SlidingWindowSplitter(initial_train_window=0.5, step=0.2)
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

    properties = InvertibleTransformation.Properties(requires_X=False)
    name = "Difference"

    def __init__(self, lag: int = 1) -> None:
        self.lag = lag

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.last_rows_X = X.iloc[-self.lag : None]

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if len(X) >= self.lag:
            self.last_rows_X = X.iloc[-self.lag : None]
        else:
            self.last_rows_X = pd.concat([self.last_rows_X, X], axis="index").iloc[
                -self.lag : None
            ]

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if in_sample:
            return X.diff(self.lag)
        else:
            return (
                pd.concat([self.last_rows_X, X], axis="index")
                .diff(self.lag)
                .iloc[self.lag :]
            )

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        return X.cumsum() + self.last_rows_X.iloc[0].squeeze()
