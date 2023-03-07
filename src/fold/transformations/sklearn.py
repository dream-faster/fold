from typing import Optional

import pandas as pd

from .base import FeatureSelector, Transformation


class SKLearnTransformation(Transformation):
    properties = Transformation.Properties()

    def __init__(self, transformation) -> None:
        if hasattr(transformation, "set_output"):
            transformation = transformation.set_output(transform="pandas")
        self.transformation = transformation
        self.name = transformation.__class__.__name__

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.transformation.fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.transformation, "set_output"):
            return self.transformation.transform(X)
        else:
            return pd.DataFrame(self.transformation.transform(X), columns=X.columns)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.transformation.inverse_transform(X), columns=X.columns, index=X.index
        )


class SKLearnFeatureSelector(FeatureSelector):
    properties = Transformation.Properties()

    def __init__(self, transformation) -> None:
        if hasattr(transformation, "set_output"):
            transformation = transformation.set_output(transform="pandas")
        self.transformation = transformation
        self.name = transformation.__class__.__name__

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.transformation.fit(X, y)
        if hasattr(self.transformation, "get_feature_names_out"):
            self.selected_features = self.transformation.get_feature_names_out()
        else:
            self.selected_features = X.columns[
                self.transformation.get_support()
            ].to_list()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features]
