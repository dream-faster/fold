from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor

from fold.base import Tunable
from fold.models.base import TimeSeriesModel


class AR(TimeSeriesModel, Tunable):
    def __init__(self, p: int, params_to_try: Optional[dict] = None) -> None:
        self.p = p
        self.name = f"AR-{str(p)}"
        self.properties = TimeSeriesModel.Properties(
            requires_X=False,
            mode=TimeSeriesModel.Properties.Mode.online,
            model_type=TimeSeriesModel.Properties.ModelType.regressor,
            memory_size=p,
            _internal_supports_minibatch_backtesting=True,
        )
        self.models = [LinearRegression() for _ in range(p)]
        self.params_to_try = params_to_try

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        # Using Least Squares as it's faster than SGD for the initial fit
        for index, model in enumerate(self.models, start=1):
            model.fit(
                y.shift(index).to_frame()[index:],
                y[index:],
                sample_weight=sample_weights[index:]
                if sample_weights is not None
                else None,
            )
        self.parameters = [
            {
                "coef_": model.coef_[0],
                "intercept_": model.intercept_,
            }
            for model in self.models
        ]

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if isinstance(self.models[0], LinearRegression):
            # If the model is being updated for the first time, convert to SGDRegressor (with the parameters from the initial fit)
            self.models = [SGDRegressor(warm_start=True) for _ in range(self.p)]
            for index, (model, parameters) in enumerate(
                zip(self.models, self.parameters), start=1
            ):
                model.fit(
                    y.shift(index).to_frame()[index:],
                    y[index:],
                    coef_init=parameters["coef_"],
                    intercept_init=parameters["intercept_"],
                    sample_weight=sample_weights[index:]
                    if sample_weights is not None
                    else None,
                )
        else:
            # For any subsequent updates, just call partial_fit
            for index, model in enumerate(self.models, start=1):
                model.partial_fit(
                    y.shift(index).to_frame()[index:],
                    y[index:],
                    sample_weight=sample_weights[index:]
                    if sample_weights is not None
                    else None,
                )

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return _predict(self.models, past_y, indices=X.index).fillna(0.0)

    predict_in_sample = predict

    def get_params(self) -> dict:
        return {"p": self.p}


def _predict(models, past_y: pd.Series, indices) -> pd.Series:
    preds = [
        np.concatenate(
            [
                np.zeros((index,)),
                lr.predict(past_y.shift(index - 1).to_frame()[index:]),
            ]
        )
        for index, lr in enumerate(models, start=1)
    ]
    return pd.Series(np.vstack(preds).mean(axis=0), index=indices)
