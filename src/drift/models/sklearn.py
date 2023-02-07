import pandas as pd

from drift.models.base import Model
from drift.transformations.base import Transformation


class SKLearnModel(Model):

    properties = Transformation.Properties(requires_past_X=False)

    def __init__(self, model) -> None:
        self.model = model
        self.name = model.__class__.__name__

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(data=self.model.predict(X).squeeze(), index=X.index)
