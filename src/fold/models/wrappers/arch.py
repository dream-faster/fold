from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd

from fold.base import fit_noop
from fold.models.base import Model
from fold.utils.checks import is_X_available


class WrapArch(Model):
    def __init__(
        self,
        init_args: dict,
        use_exogenous: Optional[bool] = None,
        online_mode: bool = False,
        instance: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> None:
        self.init_args = init_args
        init_args = {} if init_args is None else init_args
        self.use_exogenous = use_exogenous
        self.properties = Model.Properties(
            requires_X=False,
            model_type=Model.Properties.ModelType.regressor,
            mode=(
                Model.Properties.Mode.online
                if online_mode
                else Model.Properties.Mode.minibatch
            ),
        )
        self.name = name or "Arch"
        self.instance = instance

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        from arch import arch_model

        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            self.model = arch_model(y, x=X, **self.init_args)
        else:
            self.model = arch_model(y, **self.init_args)
        self.model = self.model.fit(disp="off")

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            res = self.model.forecast(horizon=len(X), reindex=False, x=X)
        else:
            res = self.model.forecast(horizon=len(X), reindex=False)
        return pd.Series(res.variance.values[0], index=X.index)

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        res = self.model.forecast(horizon=len(X), start=0, reindex=True)
        return res.variance[res.variance.columns[0]]

    update = fit_noop
