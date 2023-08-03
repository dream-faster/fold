# from __future__ import annotations

# from copy import deepcopy
# from typing import Any, Dict, Optional, Type, Union

# import numpy as np
# import pandas as pd

# from fold.models.base import Model


# class WrapNeuralForecast(Model):
#     def __init__(
#         self,
#         model_class: Type,
#         init_args: Optional[Dict],
#         instance: Optional[Any] = None,
#     ) -> None:
#         self.init_args = init_args
#         init_args = {} if init_args is None else init_args
#         self.model = model_class(**init_args) if instance is None else instance
#         self.model_class = model_class
#         self.name = self.model_class.__class__.__name__
#         self.properties = Model.Properties(
#             requires_X=False,
#             model_type=Model.Properties.ModelType.regressor,
#         )

# @classmethod
# def from_model(
#     cls,
#     model,
# ) -> WrapNeuralForecast:
#     return cls(
#         model_class=None,
#         init_args=None,
#         instance=model,
#     )

# def fit(
#     self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
# ) -> None:
#     from neuralforecast import NeuralForecast

#     self.nf = NeuralForecast(models=[self.model], freq=X.index.freqstr)
#     assert type(self.model.h) is int, "Forecasting horizon/step must be an integer."

#     data = y.rename("y").to_frame()
#     data["ds"] = X.index
#     data["unique_id"] = 1.0
#     self.nf.fit(data)

# def update(
#     self,
#     X: pd.DataFrame,
#     y: Optional[pd.Series],
#     sample_weights: Optional[pd.Series] = None,
# ) -> None:
#     for model in self.nf.models:
#         model.max_steps = 10
#     data = y.rename("y").to_frame()
#     data["ds"] = X.index
#     data["unique_id"] = 1.0
#     self.nf.fit(data)

# def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
#     predicted = self.nf.predict()

#     if len(predicted) != len(X):
#         raise ValueError(
#             "Step size (of the Splitter) and `h` (forecasting horizon) must be"
#             " equal."
#         )
#     else:
#         return pd.Series(
#             predicted[self.model.__class__.__name__].values, index=X.index
#         )

# def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
#     data = pd.DataFrame(
#         {"ds": X.index, "y": 0.0, "unique_id": 1.0},
#     )
#     nf = deepcopy(self.nf)
#     predictions = nf.predict_rolled(
#         data,
#         insample_size=len(X) - self.model.input_size,
#         n_windows=None,
#         step_size=self.model.input_size,
#     )[self.model.__class__.__name__]
#     # NeuralForecast will not return in sample predictions for `input_size`, so let's pad that with NaNs
#     padding_size = len(X) - len(predictions)
#     return pd.Series(
#         np.hstack([np.full(padding_size, np.nan), predictions.values]),
#         index=X.index,
#     )
