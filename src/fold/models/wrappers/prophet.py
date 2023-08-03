from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pandas as pd

from fold.models.base import Model


class WrapProphet(Model):
    def __init__(
        self,
        model_class: Type,
        init_args: Optional[Dict],
        online_mode: bool = False,
        instance: Optional[Any] = None,
    ) -> None:
        self.init_args = init_args
        init_args = {} if init_args is None else init_args
        self.model = model_class(**init_args) if instance is None else instance
        self.model_class = model_class
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
        online_mode: bool = False,
    ) -> WrapProphet:
        return cls(
            model_class=model.__class__,
            init_args={},
            instance=model,
            online_mode=online_mode,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        data = pd.DataFrame({"ds": X.index, "y": y.values})
        self.model.fit(data)

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if self.init_args is None:
            raise ValueError(
                "Cannot update model if init_args is None, probably .from_model"
                " constructor was used."
            )
        data = pd.DataFrame({"ds": X.index, "y": y.values})
        old_model = self.model
        self.model = self.model_class(**self.init_args)
        self.model.fit(data, init=warm_start_params(old_model))

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = pd.DataFrame({"ds": X.index})
        predictions = self.model.predict(data)
        return pd.Series(predictions["yhat"].values, index=X.index)

    predict_in_sample = predict


def warm_start_params(m):
    """
    # from https://facebook.github.io/prophet/docs/additional_topics.html
    Retrieve parameters from a trained model in the format used to initialize a new Stan model.
    Note that the new Stan model must have these same settings:
        n_changepoints, seasonality features, mcmc sampling
    for the retrieved parameters to be valid for the new model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    """
    res = {}
    for pname in ["k", "m", "sigma_obs"]:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0][0]
        else:
            res[pname] = np.mean(m.params[pname])
    for pname in ["delta", "beta"]:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0]
        else:
            res[pname] = np.mean(m.params[pname], axis=0)
    return res
