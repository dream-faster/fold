from __future__ import annotations

from typing import Union

import pandas as pd

from ..transformations.base import Transformation, fit_noop
from .base import Model


class Naive(Model):
    """
    A model that predicts the last value seen.
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
