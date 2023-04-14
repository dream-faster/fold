from __future__ import annotations

from typing import Union

import pandas as pd

from ..base import Transformation, fit_noop
from .base import Model


class Naive(Model):
    """
    A univariate model that predicts the last target value.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.models import Naive
        >>> from fold.utils.tests import generate_sine_wave_data
        >>> X, y  = generate_sine_wave_data()
        >>> splitter = SlidingWindowSplitter(initial_train_window=0.5, step=0.2)
        >>> pipeline = Naive()
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
        >>> pd.concat([preds, y[preds.index]], axis=1).head()
                             predictions_Naive    sine
        2021-12-31 15:40:00            -0.0000  0.0126
        2021-12-31 15:41:00             0.0126  0.0251
        2021-12-31 15:42:00             0.0251  0.0377
        2021-12-31 15:43:00             0.0377  0.0502
        2021-12-31 15:44:00             0.0502  0.0628
    """

    name = "Naive"
    properties = Model.Properties(
        requires_X=False,
        mode=Transformation.Properties.Mode.online,
        memory_size=1,
        _internal_supports_minibatch_backtesting=False,
    )

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            self._state.memory_y.iloc[-1].squeeze(), index=X.index[-1:None]
        )

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self._state.memory_y.shift(1)

    fit = fit_noop
    update = fit
