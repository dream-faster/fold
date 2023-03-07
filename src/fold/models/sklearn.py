from typing import Optional, Union

import pandas as pd

from fold.models.base import Model
from fold.transformations.base import Transformation


class SKLearnClassifier(Model):
    properties = Transformation.Properties(
        model_type=Transformation.Properties.ModelType.classifier
    )
    fitted = False

    def __init__(self, model) -> None:
        self.model = model
        self.name = model.__class__.__name__

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
            return
        if self.fitted:
            return
        self.model.fit(X, y, sample_weights)
        self.fitted = True

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        probabilities = pd.DataFrame(
            data=self.model.predict_proba(X),
            index=X.index,
            columns=[
                f"probabilities_{self.name}_{item}" for item in self.model.classes_
            ],
        )
        predictions = pd.Series(
            data=self.model.predict(X).squeeze(),
            index=X.index,
            name=f"predictions_{self.name}",
        )
        return pd.concat([predictions, probabilities], axis="columns")


class SKLearnRegressor(Model):
    properties = Transformation.Properties(
        model_type=Transformation.Properties.ModelType.regressor
    )
    fitted = False

    def __init__(self, model) -> None:
        self.model = model
        self.name = model.__class__.__name__

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
            return
        if self.fitted:
            return
        self.model.fit(X, y, sample_weights)
        self.fitted = True

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(
            data=self.model.predict(X).squeeze(),
            index=X.index,
            name=f"predictions_{self.name}",
        ).to_frame()


class SKLearnPipeline(Model):
    properties = Transformation.Properties()

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.name = pipeline.__class__.__name__

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.Series(
            data=self.pipeline.predict(X).squeeze(),
            index=X.index,
            name=f"predictions_{self.name}",
        ).to_frame()
