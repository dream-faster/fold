from __future__ import annotations

from typing import Any, Optional, Type, Union

import pandas as pd

from fold.models.base import Model
from fold.utils.checks import is_X_available


class WrapStatsModels(Model):
    def __init__(
        self,
        model_class: Type,
        init_args: dict,
        use_exogenous: Optional[bool] = None,
        online_mode: bool = False,
        instance: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> None:
        self.init_args = init_args
        init_args = {} if init_args is None else init_args
        self.model_class = model_class
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
        self.name = name or self.model_class.__class__.__name__
        self.instance = instance

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            self.model = (
                self.model_class(y, X, **self.init_args)
                if self.instance is None
                else self.instance
            )
            self.model = self.model.fit()
        else:
            self.model = (
                self.model_class(y, **self.init_args)
                if self.instance is None
                else self.instance
            )
            self.model = self.model.fit()

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if not hasattr(self.model, "append"):
            return
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            self.model = self.model.append(endog=y, exog=X, refit=True)
        else:
            self.model = self.model.append(endog=y, refit=True)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            return pd.Series(
                self.model.predict(start=X.index[0], end=X.index[-1], exog=X)
            )
        else:
            return pd.Series(self.model.predict(start=X.index[0], end=X.index[-1]))

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self.model.predict(start=X.index[0], end=X.index[-1])
