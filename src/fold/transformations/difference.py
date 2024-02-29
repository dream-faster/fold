# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from finml_utils.enums import ParsableEnum
from finml_utils.returns import to_log_returns, to_returns

from ..base import Transformation, Tunable, fit_noop


def __handle_close_to_zero_returns(prices: pd.Series) -> pd.Series:
    res_prices = pd.Series(index=prices.index)

    in_range = ((prices < 1).astype(int) + (prices > -1).astype(int)) == 2
    in_range = in_range.shift(1).fillna(False)

    res_prices[in_range] = prices.diff()[in_range]
    res_prices[~in_range] = to_returns(prices)[~in_range]

    return res_prices


class StationaryMethod(ParsableEnum):
    difference = "difference"
    log_returns = "log_returns"
    returns = "returns"
    capped_returns = "capped_returns"

    def get_transform_func(self) -> Callable:
        if self == StationaryMethod.difference:
            return lambda x: x.diff()
        if self == StationaryMethod.log_returns:
            return lambda x: to_log_returns(x)
        if self == StationaryMethod.returns:
            return lambda x: to_returns(x)
        if self == StationaryMethod.capped_returns:
            return lambda x: __handle_close_to_zero_returns(x)
        raise ValueError(f"Unknown TransformationMethod: {self}")


class Difference(Transformation, Tunable):
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
    >>> from fold.transformations import Difference
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data(freq="min")
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = Difference()
    >>> preds, trained_pipeline, _, _ = train_backtest(pipeline, X, y, splitter)
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

    def __init__(
        self,
        method: StationaryMethod | str,
        clip_threshold: float | None = None,
        fill_0_with_last: bool = False,
        name: str | None = None,
        params_to_try: dict | None = None,
    ) -> None:
        self.method = StationaryMethod.from_str(method)
        self.params_to_try = params_to_try
        self.clip_threshold = clip_threshold
        self.fill_0_with_last = fill_0_with_last
        self.name = name or f"Difference-{self.method.value}"
        self.properties = Transformation.Properties(requires_X=True)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        rets = self.method.get_transform_func()(X)
        if self.clip_threshold is not None:
            rets = rets.clip(lower=-self.clip_threshold, upper=self.clip_threshold)
        if self.fill_0_with_last:
            rets = rets.replace(0.0, pd.NA).ffill()
        return rets

    fit = fit_noop
    update = fit_noop
