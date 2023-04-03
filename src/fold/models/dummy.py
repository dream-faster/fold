from typing import Union

import pandas as pd

from ..base import Transformation, fit_noop
from .base import Model


class DummyClassifier(Model):
    """
    A model that predicts a single class and probabilities.
    """

    properties = Transformation.Properties(requires_X=False)
    name = "DummyClassifier"

    def __init__(self, predicted_value, all_classes, predicted_probabilities) -> None:
        self.predicted_value = predicted_value
        self.all_classes = all_classes
        self.predicted_probabilities = predicted_probabilities

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        predictions = pd.Series(
            [self.predicted_value] * len(X),
            index=X.index,
            name="predictions_DummyClassifier",
        )
        probabilities = [
            pd.Series(
                [prob] * len(X),
                index=X.index,
                name=f"probabilities_DummyClassifier_{associated_class}",
            )
            for associated_class, prob in zip(
                self.all_classes, self.predicted_probabilities
            )
        ]

        return pd.concat([predictions] + probabilities, axis="columns")

    predict_in_sample = predict
    fit = fit_noop
    update = fit


class DummyRegressor(Model):
    """
    A model that predicts a single value.
    """

    properties = Transformation.Properties(requires_X=False)
    name = "DummyRegressor"

    def __init__(self, predicted_value: float) -> None:
        self.predicted_value = predicted_value

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(
            [self.predicted_value] * len(X),
            index=X.index,
        )

    predict_in_sample = predict
    fit = fit_noop
    update = fit
