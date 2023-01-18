import pandas as pd

from .base import FeatureSelector, Transformation


class SKLearnTransformation(Transformation):
    def __init__(self, model) -> None:
        self.model = model.set_output(transform="pandas")
        self.name = model.__class__.__name__

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)


class SKLearnFeatureSelector(FeatureSelector):
    def __init__(self, model) -> None:
        self.model = model.set_output(transform="pandas")
        self.name = model.__class__.__name__

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.selected_features = self.model.get_feature_names_out()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features]
