import pandas as pd

from drift.models.base import Model


class SKLearnModel(Model):

    name = "SKLearnModel"

    def __init__(self, model) -> None:
        self.model = model.set_output(transform="pandas")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)
