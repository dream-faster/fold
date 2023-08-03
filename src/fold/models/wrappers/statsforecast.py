from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

import pandas as pd

from fold.models.base import Model
from fold.utils.checks import is_X_available


class WrapStatsForecast(Model):
    def __init__(
        self,
        model_class: Type,
        init_args: Optional[Dict],
        use_exogenous: Optional[bool] = None,
        online_mode: bool = False,
        instance: Optional[Any] = None,
    ) -> None:
        self.init_args = init_args
        init_args = {} if init_args is None else init_args
        self.model = model_class(**init_args) if instance is None else instance
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
        self.name = self.model_class.__class__.__name__

    @classmethod
    def from_model(
        cls,
        model,
        use_exogenous: Optional[bool] = None,
        online_mode: bool = False,
    ) -> WrapStatsForecast:
        return cls(
            model_class=model.__class__,
            init_args={},
            use_exogenous=use_exogenous,
            instance=model,
            online_mode=online_mode,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            self.model.fit(y=y.values, X=X.values)
        else:
            self.model.fit(y=y.values)

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if not hasattr(self.model, "forward"):
            return
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            self.model.forward(y=y.values, h=len(X), X=X.values)
        else:
            self.model.forward(y=y.values, h=len(X))

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            return pd.Series(
                self.model.predict(h=len(X), X=X.values)["mean"], index=X.index
            )
        else:
            return pd.Series(self.model.predict(h=len(X))["mean"], index=X.index)

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        pred_dict = self.model.predict_in_sample()
        if "fitted" in pred_dict:
            return pd.Series(pred_dict["fitted"], index=X.index)
        elif "mean" in pred_dict:
            return pd.Series(pred_dict["mean"], index=X.index)
        else:
            raise ValueError("Unknown prediction dictionary structure")
