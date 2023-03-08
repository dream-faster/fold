from typing import Optional, Union

import pandas as pd

from fold.models.base import Model
from fold.transformations.base import Transformation


class SKLearnClassifier(Model):
    properties = Model.Properties(model_type=Model.Properties.ModelType.classifier)

    def __init__(self, model) -> None:
        self.model = model
        self.name = model.__class__.__name__

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
        else:
            self.model.fit(X, y, sample_weights)

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
        # if we don't have partial_fit, we can't update the model (maybe throw an exception, and force user to wrap it into `DontUpdate`?)

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

    predict_in_sample = predict


class SKLearnRegressor(Model):
    properties = Model.Properties(model_type=Model.Properties.ModelType.regressor)

    def __init__(self, model) -> None:
        self.model = model
        self.name = model.__class__.__name__

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
        else:
            self.model.fit(X, y, sample_weights)

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
        # if we don't have partial_fit, we can't update the model (maybe throw an exception, and force user to wrap it into `DontUpdate`?)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(
            data=self.model.predict(X).squeeze(),
            index=X.index,
            name=f"predictions_{self.name}",
        ).to_frame()

    predict_in_sample = predict


class SKLearnPipeline(Model):
    properties = Transformation.Properties()

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.name = pipeline.__class__.__name__

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.Series(
            data=self.pipeline.predict(X).squeeze(),
            index=X.index,
            name=f"predictions_{self.name}",
        ).to_frame()

    predict_in_sample = predict
